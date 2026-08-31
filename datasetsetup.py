"""Prepare SWED images and masks for training a segmentation model.

Samples from the ``train`` directory are paired, validated, split by scene
into train/validation/test subsets, and converted into six-band PyTorch datasets.
"""

from collections.abc import Callable
from itertools import combinations
import logging
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "Sentinel-2WaterDataset(SWED)" / "SWED"
TRAIN_IMAGES_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LABELS_DIR = DATASET_ROOT / "train" / "labels"

# Fractions applied to the single SWED/train data source.
TRAIN_SPLIT_FRACTION = 0.50
VALIDATION_SPLIT_FRACTION = 0.30
TEST_SPLIT_FRACTION = 0.20

IMAGE_SIZE = (256, 256)
SUPPORTED_EXTENSIONS = {".npy", ".tif", ".tiff"}

# SWED order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12.
# These indices select Blue, Green, Red, NIR, SWIR1, and SWIR2.
SIX_BAND_INDICES = (1, 2, 3, 7, 10, 11)
REFLECTANCE_SCALE = 8160.0

TRAIN_TRANSFORM = v2.Compose(
    [
        # Flip the image and mask horizontally with 50% probability.
        v2.RandomHorizontalFlip(p=0.5),
        # Flip the image and mask vertically with 50% probability.
        v2.RandomVerticalFlip(p=0.5),
        # Uniformly choose one of four rotations, each with 25% probability.
        # Identity corresponds to a 0-degree rotation.
        v2.RandomChoice(
            [
                v2.Identity(),
                v2.RandomRotation((90, 90)),
                v2.RandomRotation((180, 180)),
                v2.RandomRotation((270, 270)),
            ]
        ),
        # Add Gaussian noise only to the image with 50% probability.
        # mean=0 avoids a systematic brightness shift, sigma=0.01 controls
        # noise intensity, and clip=True keeps values between 0 and 1.
        v2.RandomApply(
            [v2.GaussianNoise(mean=0.0, sigma=0.01, clip=True)],
            p=0.5,
        ),
    ]
)


class _IgnoreInvalidGDALNoDataTag(logging.Filter):
    """Hide only the harmless warning related to GDAL_NODATA."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Decide whether a log message should be displayed.

        Args:
            record: Message produced by the tifffile logger.

        Returns:
            ``False`` for the ignored GDAL_NODATA warning; otherwise ``True``.
        """
        return "parsing GDAL_NODATA tag raised" not in record.getMessage()


_TIFFFILE_LOG_FILTER = _IgnoreInvalidGDALNoDataTag()


def _sample_id(path: Path, markers: tuple[str, ...]) -> str | None:
    """Extract the common identifier used to pair an image with its mask.

    Args:
        path: File path from which the identifier is extracted.
        markers: Accepted filename markers, such as ``_image_``.

    Returns:
        Normalized identifier, or ``None`` when no marker is found.
    """
    for marker in markers:
        if marker in path.stem:
            return path.stem.replace(marker, "_", 1)
    return None


def _index_files(directory: Path, markers: tuple[str, ...]) -> dict[str, Path]:
    """Index supported files by their normalized identifier.

    Args:
        directory: Directory containing images or masks.
        markers: Markers recognized in filenames.

    Returns:
        Mapping from each identifier to its file path.
    """
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        sample_id = _sample_id(path, markers)
        if sample_id is None:
            continue
        if sample_id in indexed:
            raise ValueError(f"ID duplicato '{sample_id}' nella cartella {directory}")
        indexed[sample_id] = path
    return indexed


def get_pairs(
    images_dir: str | Path,
    labels_dir: str | Path,
) -> pd.DataFrame:
    """Pair images with masks and remove pairs containing invalid masks.

    Args:
        images_dir: Directory containing satellite images.
        labels_dir: Directory containing the corresponding masks.

    Returns:
        DataFrame containing IDs, paths, and pixel statistics for each pair.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Cartella immagini non trovata: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Cartella label non trovata: {labels_dir}")

    images = _index_files(images_dir, ("_image_",))
    labels = _index_files(labels_dir, ("_chip_", "_label_"))
    if not images:
        raise ValueError(f"Nessuna immagine supportata in {images_dir}")
    if not labels:
        raise ValueError(f"Nessuna label supportata in {labels_dir}")

    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    if missing_labels or missing_images:
        raise ValueError(
            "Sono presenti file senza coppia. "
            f"Immagini senza label: {missing_labels[:5]}; "
            f"label senza immagine: {missing_images[:5]}"
        )

    pairs = pd.DataFrame(
        {
            "sample_id": sample_id,
            "image_path": images[sample_id],
            "label_path": labels[sample_id],
        }
        for sample_id in sorted(images)
    )
    return filter_invalid_masks(pairs)


def split_pairs(
    pairs: pd.DataFrame,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split pairs by scene while preserving water-pixel prevalence.

    Args:
        pairs: Pairs containing ``sample_id`` and pixel statistics.
        train_fraction: Fraction assigned to the first returned split.
        seed: Seed used for deterministic tie-breaking.

    Returns:
        First and second split DataFrames, with no shared scenes.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction deve essere compreso tra 0 e 1")
    if "sample_id" not in pairs:
        raise ValueError("Il DataFrame deve contenere la colonna 'sample_id'")

    frame = pairs.copy()
    # The final two ID fields represent the tile row and column.
    frame["scene_id"] = frame["sample_id"].str.rsplit("_", n=2).str[0]
    if (frame["scene_id"] == frame["sample_id"]).any():
        raise ValueError("Un sample_id non termina con '_riga_colonna'")

    scenes = sorted(frame["scene_id"].unique())
    if len(scenes) < 2:
        raise ValueError("Servono almeno due scene per creare train e validation")
    split_at = min(max(round(len(scenes) * train_fraction), 1), len(scenes) - 1)
    validation_scene_count = len(scenes) - split_at

    required_statistics = {"water_pixels", "total_pixels"}
    if not required_statistics.issubset(frame.columns):
        raise ValueError(
            "Lo split richiede le colonne 'water_pixels' e 'total_pixels'"
        )

    # Select complete scenes that match both the desired size and global water
    # prevalence while preventing leakage between tiles from the same scene.
    scene_stats = frame.groupby("scene_id")[["water_pixels", "total_pixels"]].sum()
    scene_sample_counts = frame.groupby("scene_id").size()
    target_samples = len(frame) * (1.0 - train_fraction)
    target_water_fraction = frame["water_pixels"].sum() / frame["total_pixels"].sum()
    shuffled_scenes = scenes.copy()
    random.Random(seed).shuffle(shuffled_scenes)

    best_score = float("inf")
    validation_scenes: set[str] = set()
    for candidate in combinations(shuffled_scenes, validation_scene_count):
        candidate_names = list(candidate)
        candidate_samples = int(scene_sample_counts.loc[candidate_names].sum())
        candidate_stats = scene_stats.loc[candidate_names].sum()
        size_error = abs(candidate_samples - target_samples) / len(frame)
        water_fraction = (
            candidate_stats["water_pixels"] / candidate_stats["total_pixels"]
        )
        prevalence_error = abs(water_fraction - target_water_fraction)
        score = float(size_error + prevalence_error)
        if score < best_score:
            best_score = score
            validation_scenes = set(candidate)
    train_scenes = set(scenes) - validation_scenes

    train = frame[frame["scene_id"].isin(train_scenes)].reset_index(drop=True)
    validation = frame[~frame["scene_id"].isin(train_scenes)].reset_index(drop=True)
    return train, validation


def split_pairs_train_validation_test(
    pairs: pd.DataFrame,
    train_fraction: float = TRAIN_SPLIT_FRACTION,
    validation_fraction: float = VALIDATION_SPLIT_FRACTION,
    test_fraction: float = TEST_SPLIT_FRACTION,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create scene-disjoint splits with consistent water/land distribution.

    Args:
        pairs: Pairs containing identifiers and pixel statistics.
        train_fraction: Overall fraction assigned to training.
        validation_fraction: Overall fraction assigned to validation.
        test_fraction: Overall fraction assigned to testing.
        seed: Seed that makes the split reproducible.

    Returns:
        Train, validation, and test DataFrames, in this order.
    """
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(fraction <= 0.0 for fraction in fractions):
        raise ValueError("Le frazioni train, validation e test devono essere positive")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Le frazioni train, validation e test devono sommare a 1")

    train, holdout = split_pairs(
        pairs,
        train_fraction=train_fraction,
        seed=seed,
    )
    validation_share_of_holdout = validation_fraction / (
        validation_fraction + test_fraction
    )
    validation, test = split_pairs(
        holdout,
        train_fraction=validation_share_of_holdout,
        seed=seed + 1,
    )

    split_frames = {"train": train, "validation": validation, "test": test}
    scene_sets = {
        name: set(frame["scene_id"])
        for name, frame in split_frames.items()
    }
    if (
        scene_sets["train"] & scene_sets["validation"]
        or scene_sets["train"] & scene_sets["test"]
        or scene_sets["validation"] & scene_sets["test"]
    ):
        raise RuntimeError("Spatial leakage: una scena compare in più split")

    return train, validation, test


def filter_invalid_masks(pairs: pd.DataFrame) -> pd.DataFrame:
    """Keep binary 256x256 masks and calculate per-class pixel statistics.

    Args:
        pairs: DataFrame containing at least the ``label_path`` column.

    Returns:
        Valid pairs with ``water_pixels`` and ``total_pixels`` columns.
    """
    valid_rows: list[bool] = []
    water_pixels: list[int] = []
    total_pixels: list[int] = []
    for label_path in pairs["label_path"]:
        mask = np.squeeze(SWEDDataset._load_array(label_path))
        is_valid = mask.shape == IMAGE_SIZE and np.isin(mask, (0, 1)).all()
        valid_rows.append(is_valid)
        water_pixels.append(int(np.count_nonzero(mask == 1)) if is_valid else 0)
        total_pixels.append(int(mask.size) if is_valid else 0)

    annotated = pairs.assign(
        water_pixels=water_pixels,
        total_pixels=total_pixels,
    )
    valid = annotated.loc[valid_rows].reset_index(drop=True)
    discarded = len(pairs) - len(valid)
    if discarded:
        print(f"Maschere non valide escluse: {discarded}/{len(pairs)}")
    if valid.empty:
        raise ValueError("Nessuna coppia con maschera binaria valida")
    return valid


def normalize_six_bands(image: np.ndarray) -> Tensor:
    """Select and normalize the six bands used by the model.

    Args:
        image: Raw 12-band SWED array in HWC or CHW format.

    Returns:
        Float32 tensor shaped ``[6, 256, 256]`` with values between 0 and 1.
    """
    if image.ndim != 3:
        raise ValueError(f"Immagine a 3 dimensioni attesa, ricevuta {image.shape}")

    # Accept both dimension orders found in the supported files.
    if image.shape[-1] == 12:
        image_hwc = image
    elif image.shape[0] == 12:
        image_hwc = np.moveaxis(image, 0, -1)
    else:
        raise ValueError(f"Immagine a 12 bande attesa, ricevuta {image.shape}")
    if image_hwc.shape[:2] != IMAGE_SIZE:
        raise ValueError(f"Immagine {IMAGE_SIZE} attesa, ricevuta {image_hwc.shape[:2]}")
    selected = image_hwc[..., SIX_BAND_INDICES].astype(np.float32, copy=False)
    selected = np.clip(selected / REFLECTANCE_SCALE, 0.0, 1.0)
    return torch.from_numpy(selected).permute(2, 0, 1).contiguous()


def random_training_transform(image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Apply synchronized random augmentation to an image and its mask.

    Args:
        image: Normalized six-band image.
        mask: Binary mask associated with the image.

    Returns:
        Transformed image and geometrically aligned mask.
    """
    # The Mask type synchronizes geometric transforms and excludes the mask
    # from image-only transforms such as noise.
    image, mask = TRAIN_TRANSFORM(image, tv_tensors.Mask(mask))
    return image.contiguous(), mask.as_subclass(torch.Tensor).contiguous()


class SWEDDataset(Dataset[tuple[Tensor, Tensor]]):
    """PyTorch dataset that lazily loads SWED images and binary masks."""

    def __init__(
        self,
        pairs: pd.DataFrame,
        transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            pairs: DataFrame containing image and mask paths.
            transform: Optional function jointly applied to the pair.
        """
        required = {"image_path", "label_path"}
        if missing := required - set(pairs.columns):
            raise ValueError(f"Colonne mancanti nel DataFrame: {sorted(missing)}")
        self.pairs = pairs.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of image-mask pairs in the dataset."""
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Load and prepare one sample.

        Args:
            index: Position of the requested sample.

        Returns:
            Six-band image tensor and binary mask tensor.
        """
        # Load data on demand instead of keeping the entire dataset in memory.
        row = self.pairs.iloc[index]
        image = normalize_six_bands(self._load_array(row["image_path"]))
        mask_array = np.squeeze(self._load_array(row["label_path"]))

        if mask_array.shape != IMAGE_SIZE:
            raise ValueError(
                f"Maschera {IMAGE_SIZE} attesa, ricevuta {mask_array.shape} "
                f"in {row['label_path']}"
            )
        if not np.isin(mask_array, (0, 1)).all():
            raise ValueError(f"La maschera non è binaria: {row['label_path']}")
        # CrossEntropyLoss expects class indices in int64/long format.
        mask = torch.from_numpy(mask_array.astype(np.int64, copy=False))

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask

    @staticmethod
    def _load_array(path: str | Path) -> np.ndarray:
        """Read a supported NumPy or TIFF file.

        Args:
            path: Path to the ``.npy``, ``.tif``, or ``.tiff`` file.

        Returns:
            File contents as a NumPy array.
        """
        path = Path(path)
        if path.suffix.lower() == ".npy":
            return np.load(path, allow_pickle=False)
        if path.suffix.lower() in {".tif", ".tiff"}:
            import tifffile

            logger = logging.getLogger("tifffile")
            logger.addFilter(_TIFFFILE_LOG_FILTER)
            try:
                return tifffile.imread(path)
            finally:
                logger.removeFilter(_TIFFFILE_LOG_FILTER)
        raise ValueError(f"Formato non supportato: {path}")


def create_datasets(
    seed: int = 42,
    train_images_dir: str | Path = TRAIN_IMAGES_DIR,
    train_labels_dir: str | Path = TRAIN_LABELS_DIR,
    train_fraction: float = TRAIN_SPLIT_FRACTION,
    validation_fraction: float = VALIDATION_SPLIT_FRACTION,
    test_fraction: float = TEST_SPLIT_FRACTION,
) -> tuple[SWEDDataset, SWEDDataset, SWEDDataset]:
    """Build the three PyTorch datasets from the SWED/train directory.

    Args:
        seed: Seed used for reproducible splitting.
        train_images_dir: Directory containing source images.
        train_labels_dir: Directory containing source masks.
        train_fraction: Fraction assigned to training.
        validation_fraction: Fraction assigned to validation.
        test_fraction: Fraction assigned to testing.

    Returns:
        Training, validation, and test datasets, in this order.
    """
    # Remove invalid masks before performing any split.
    development_pairs = get_pairs(train_images_dir, train_labels_dir)
    train_pairs, validation_pairs, test_pairs = split_pairs_train_validation_test(
        development_pairs,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    train_dataset = SWEDDataset(
        train_pairs,
        transform=random_training_transform,
    )
    validation_dataset = SWEDDataset(validation_pairs)
    test_dataset = SWEDDataset(test_pairs)
    return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, validation_dataset, test_dataset = create_datasets()
    image, mask = train_dataset[0]
    print(f"Campioni: train={len(train_dataset)}, val={len(validation_dataset)}, test={len(test_dataset)}")
    print(f"Immagine: shape={tuple(image.shape)}, dtype={image.dtype}")
    print(f"Maschera: shape={tuple(mask.shape)}, dtype={mask.dtype}")
