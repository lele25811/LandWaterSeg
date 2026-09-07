"""Train an SMP U-Net with a MobileNetV2 encoder on six Sentinel-2 bands."""

from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from train import (
    RANDOM_SEED,
    _format_metrics,
    create_dataloaders,
    run_epoch,
    save_training_curves,
)


BATCH_SIZE = 8
NUM_WORKERS = 0
INPUT_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
NUM_INPUT_CHANNELS = len(INPUT_BANDS)
NUM_CLASSES = 1
NUM_EPOCHS = 50
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
GRADIENT_ACCUMULATION_STEPS = 2  # effective batch size: 8 * 2 = 16
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5
ENCODER_NAME = "mobilenet_v2"
ENCODER_DEPTH = 4
DECODER_CHANNELS = (128, 64, 32, 16)
CHECKPOINT_PATH = Path(__file__).resolve().parent / "best_unet_mobilenet_v2_6bands.pt"
PLOT_DIRECTORY = Path(__file__).resolve().parent / "plot_curve_unet_mobilenet_v2_6bands"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UNetMobileNetV2SixBands(nn.Module):
    """Adapt an SMP binary U-Net to the common two-probability train interface."""

    def __init__(self) -> None:
        """Build the six-band U-Net after validating the decoder configuration."""
        super().__init__()
        # Each depth level requires a decoder channel count.
        if len(DECODER_CHANNELS) != ENCODER_DEPTH:
            raise ValueError(
                "decoder_channels must contain one value for each level "
                "of encoder_depth"
            )

        self.model = smp.Unet(
            encoder_name=ENCODER_NAME,  # MobileNetV2 extracts image features (encoder).
            encoder_weights=None,  # Random weights, without pretraining.
            in_channels=NUM_INPUT_CHANNELS,  # Six satellite bands as input.
            classes=NUM_CLASSES,  # One logit per pixel for binary water/land segmentation.
            encoder_depth=ENCODER_DEPTH,  # Four depth levels.
            decoder_channels=DECODER_CHANNELS,  # Channel counts for the four U-Net decoder blocks.
            activation=None,  # Raw logits; sigmoid is applied later.
        )

    def forward(
        self,
        images: Tensor,
        masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Predict land/water probabilities and optionally compute binary loss.

        Args:
            images: Input tensor shaped [batch, 6, H, W].
            masks: Optional labels shaped [batch, H, W], with 0 for land
                and 1 for water.

        Returns:
            Probabilities shaped [batch, 2, H, W], ordered as land then water,
            and a scalar loss, or None when masks are not provided.
        """
        # Expect batches of six-band images: [batch, 6, H, W].
        if images.ndim != 4 or images.shape[1] != NUM_INPUT_CHANNELS:
            raise ValueError(
                f"Expected input [batch, {NUM_INPUT_CHANNELS}, H, W], "
                f"got {tuple(images.shape)}"
            )

        water_logits = self.model(images)
        # SMP must return one value per pixel at the input spatial resolution.
        expected_shape = (images.shape[0], 1, images.shape[2], images.shape[3])
        if tuple(water_logits.shape) != expected_shape:
            raise RuntimeError(
                f"Expected SMP output {expected_shape}, "
                f"got {tuple(water_logits.shape)}"
            )

        loss = None
        # Masks are optional: without labels, return predictions only.
        if masks is not None:
            # Expect labels shaped [batch, H, W]: 0 = land, 1 = water.
            if tuple(masks.shape) != expected_shape[:1] + expected_shape[2:]:
                raise ValueError(
                    f"Expected masks {expected_shape[:1] + expected_shape[2:]}, "
                    f"got {tuple(masks.shape)}"
                )
            # This loss handles sigmoid internally for numerical stability.
            loss = F.binary_cross_entropy_with_logits(
                water_logits.squeeze(1),  # Remove the singleton channel: [batch, H, W].
                masks.float(),  # Convert labels to the floating-point type required by the loss.
            )

        # Convert logits to water probabilities between 0 and 1.
        water_probabilities = torch.sigmoid(water_logits)
        # Match the shared training interface: two channels [P(land), P(water)].
        class_probabilities = torch.cat(
            (1.0 - water_probabilities, water_probabilities),
            dim=1,
        )
        return class_probabilities, loss


def train_unet_mobilenet_v2_6bands(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> dict[str, list[dict[str, float]]]:
    """Train the U-Net and restore the checkpoint with the best validation IoU.

    Args:
        model: Model on the target device returning probabilities and loss.
        train_loader: Batches of training images and binary masks.
        validation_loader: Batches used to evaluate the model after each epoch.
        device: Device used to process training and validation batches.

    Returns:
        Per-epoch metric dictionaries under the train and validation keys.
        The supplied model is updated in place with its best saved weights.
    """
    # AdamW updates the weights and applies weight decay for regularization.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    # Reduce the learning rate when validation IoU stops improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=LEARNING_RATE / 1000,
    )
    history: dict[str, list[dict[str, float]]] = {"train": [], "validation": []}
    best_iou = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # Accumulate gradients across batches before each optimizer update.
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            description=f"Epoch {epoch}/{NUM_EPOCHS} - Train U-Net MobileNetV2 6 bands",
            accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        )
        # Evaluate without optimizer updates to monitor generalization.
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            description=(
                f"Epoch {epoch}/{NUM_EPOCHS} - Validation "
                "U-Net MobileNetV2 6 bands"
            ),
        )
        history["train"].append(train_metrics)
        history["validation"].append(validation_metrics)
        scheduler.step(validation_metrics["iou"])

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(_format_metrics("Train", train_metrics))
        print(_format_metrics("Validation", validation_metrics))
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        current_iou = validation_metrics["iou"]
        # Save improved weights, training state, and architecture metadata.
        if current_iou > best_iou:
            best_iou = current_iou
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "validation_metrics": validation_metrics,
                    "architecture": "smp.Unet",
                    "encoder_name": ENCODER_NAME,
                    "encoder_weights": None,
                    "encoder_depth": ENCODER_DEPTH,
                    "decoder_channels": DECODER_CHANNELS,
                    "input_channels": NUM_INPUT_CHANNELS,
                    "input_bands": INPUT_BANDS,
                    "num_output_classes": NUM_CLASSES,
                },
                CHECKPOINT_PATH,
            )
            print(
                f"New best model saved to {CHECKPOINT_PATH} "
                f"(IoU={best_iou:.4f})"
            )
        else:
            # Stop after the configured number of epochs without improvement.
            epochs_without_improvement += 1
            print(
                "IoU has not improved for "
                f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs"
            )
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    save_training_curves(history, PLOT_DIRECTORY)
    # Use the best validation checkpoint for subsequent test evaluation.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(
        f"Restored U-Net MobileNetV2 6 bands from epoch {checkpoint['epoch']} "
        f"with validation IoU={checkpoint['validation_metrics']['iou']:.4f}"
    )
    return history


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run train_unet_mobilenet_v2_6bands.py "
            "with an NVIDIA GPU."
        )                                                                                                                                                                                                                                                                                                   

    # Seed random generators to make runs more reproducible.
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    print(f"Device: {torch.cuda.get_device_name(DEVICE)}")
    print(
        f"Model: SMP U-Net, encoder {ENCODER_NAME}, random initialization, "
        f"bands {', '.join(INPUT_BANDS)}"
    )
    print("Dataset: train/validation/test splits derived from SWED/train")

    # Build separate splits for weight updates, model selection, and final testing.
    train_loader, validation_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=RANDOM_SEED,
    )
    print(
        f"Batch: train={len(train_loader)}, "
        f"validation={len(validation_loader)}, test={len(test_loader)}"
    )

    model = UNetMobileNetV2SixBands().to(DEVICE)
    train_unet_mobilenet_v2_6bands(
        model,
        train_loader,
        validation_loader,
        DEVICE,
    )

    # Evaluate the restored best model on the held-out test split.
    test_metrics = run_epoch(
        model,
        test_loader,
        DEVICE,
        description="Test U-Net MobileNetV2 6 bands",
    )
    print("\nFinal U-Net MobileNetV2 6 bands results on the test set")
    print(_format_metrics("Test", test_metrics))
