"""Prepare SWED images and masks for training a segmentation model.

The workflow is: match every satellite image with its mask, split the data
into training and validation sets, normalize the image bands, and create
PyTorch datasets. A separate test set is also created for final evaluation.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "Sentinel-2WaterDataset(SWED)" / "SWED"
TRAIN_IMAGES_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LABELS_DIR = DATASET_ROOT / "train" / "labels"
TEST_IMAGES_DIR = DATASET_ROOT / "test" / "images"
TEST_LABELS_DIR = DATASET_ROOT / "test" / "labels"

IMAGE_SIZE = (256, 256)
SUPPORTED_EXTENSIONS = {".npy", ".tif", ".tiff"}

# SWED: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12.
# SATLAS Sentinel2_*_SI_MS: R, G, B, B05, B06, B07, B08, B11, B12.
SATLAS_BAND_INDICES = (3, 2, 1, 4, 5, 6, 7, 10, 11)
SATLAS_BAND_NAMES = ("B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12")
SATLAS_REFLECTANCE_SCALE = 8160.0


def _sample_id(path: Path, markers: tuple[str, ...]) -> str | None:
    """Return the ID used to match an image file with its label file.

    The image and label filenames contain different markers. Replacing that
    marker with ``_`` leaves the common part of the two filenames. If none of
    the expected markers is present, return ``None``.
    """
    for marker in markers:
        if marker in path.stem:
            return path.stem.replace(marker, "_", 1)
    return None


def _index_files(directory: Path, markers: tuple[str, ...]) -> dict[str, Path]:
    """Map each valid sample ID to its file inside a directory.

    Unsupported files and files without a known marker are ignored. Duplicate
    IDs raise an error because they would make image-label pairing ambiguous.
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


def get_pairs(images_dir: str | Path, labels_dir: str | Path) -> pd.DataFrame:
    """Match every image with its label and return the pairs in a DataFrame.

    Each output row contains the sample ID, image path, and label path. Files
    are matched by the ID in their names, not by their position in a directory
    listing. An error is raised if a directory is invalid or a file has no
    matching partner.
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

    return pd.DataFrame(
        {
            "sample_id": sample_id,
            "image_path": images[sample_id],
            "label_path": labels[sample_id],
        }
        for sample_id in sorted(images)
    )


def split_pairs(
    pairs: pd.DataFrame,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide paired samples into training and validation sets.

    All tiles from the same satellite scene stay in the same set. This avoids
    evaluating the model on tiles that are very similar to its training data.
    ``train_fraction`` controls the share of scenes used for training, while
    ``seed`` makes the split reproducible.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction deve essere compreso tra 0 e 1")
    if "sample_id" not in pairs:
        raise ValueError("Il DataFrame deve contenere la colonna 'sample_id'")

    frame = pairs.copy()
    # The final two fields are the row and column of the 256x256 tile.
    frame["scene_id"] = frame["sample_id"].str.rsplit("_", n=2).str[0]
    if (frame["scene_id"] == frame["sample_id"]).any():
        raise ValueError("Un sample_id non termina con '_riga_colonna'")

    scenes = sorted(frame["scene_id"].unique())
    if len(scenes) < 2:
        raise ValueError("Servono almeno due scene per creare train e validation")
    random.Random(seed).shuffle(scenes)
    split_at = min(max(round(len(scenes) * train_fraction), 1), len(scenes) - 1)
    train_scenes = set(scenes[:split_at])

    train = frame[frame["scene_id"].isin(train_scenes)].reset_index(drop=True)
    validation = frame[~frame["scene_id"].isin(train_scenes)].reset_index(drop=True)
    return train, validation


def normalize_for_satlas(image: np.ndarray) -> Tensor:
    """Convert a raw 12-band SWED image into a SATLAS model input.

    The function accepts both H x W x C and C x H x W arrays. It selects the
    nine bands used by SATLAS, scales reflectance values to [0, 1], and returns
    a float32 tensor arranged as C x H x W.
    """
    if image.ndim != 3:
        raise ValueError(f"Immagine a 3 dimensioni attesa, ricevuta {image.shape}")

    # Training .npy files use HWC; test GeoTIFF files may use CHW.
    if image.shape[-1] == 12:
        image_hwc = image
    elif image.shape[0] == 12:
        image_hwc = np.moveaxis(image, 0, -1)
    else:
        raise ValueError(f"Immagine a 12 bande attesa, ricevuta {image.shape}")
    if image_hwc.shape[:2] != IMAGE_SIZE:
        raise ValueError(f"Immagine {IMAGE_SIZE} attesa, ricevuta {image_hwc.shape[:2]}")

    selected = image_hwc[..., SATLAS_BAND_INDICES].astype(np.float32, copy=False)
    selected = np.clip(selected / SATLAS_REFLECTANCE_SCALE, 0.0, 1.0)
    return torch.from_numpy(selected).permute(2, 0, 1).contiguous()


@dataclass(frozen=True)
class RandomGeometricTransform:
    """Randomly flip and rotate a training image together with its mask.

    Applying the same transformation to both tensors keeps every mask pixel
    aligned with the corresponding image pixel.
    """

    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    random_rotation_90: bool = True

    def __post_init__(self) -> None:
        for value in (self.horizontal_flip_probability, self.vertical_flip_probability):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Le probabilità devono essere comprese tra 0 e 1")

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        # Applying identical operations preserves pixel-to-label alignment.
        if torch.rand(()) < self.horizontal_flip_probability:
            image, mask = image.flip(-1), mask.flip(-1)
        if torch.rand(()) < self.vertical_flip_probability:
            image, mask = image.flip(-2), mask.flip(-2)
        if self.random_rotation_90:
            turns = int(torch.randint(0, 4, ()).item())
            image = torch.rot90(image, turns, dims=(-2, -1))
            mask = torch.rot90(mask, turns, dims=(-2, -1))
        return image.contiguous(), mask.contiguous()


class SWEDDataset(Dataset[tuple[Tensor, Tensor]]):
    """Provide normalized SWED images and binary masks to a DataLoader.

    Files are loaded only when a sample is requested, so the full dataset does
    not need to fit in memory. An optional transform can augment each pair.
    """

    def __init__(
        self,
        pairs: pd.DataFrame,
        transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
    ) -> None:
        required = {"image_path", "label_path"}
        if missing := required - set(pairs.columns):
            raise ValueError(f"Colonne mancanti nel DataFrame: {sorted(missing)}")
        self.pairs = pairs.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        # Lazy loading avoids keeping the entire dataset in memory.
        row = self.pairs.iloc[index]
        image = normalize_for_satlas(self._load_array(row["image_path"]))
        mask_array = np.squeeze(self._load_array(row["label_path"]))

        if mask_array.shape != IMAGE_SIZE:
            raise ValueError(
                f"Maschera {IMAGE_SIZE} attesa, ricevuta {mask_array.shape} "
                f"in {row['label_path']}"
            )
        if not np.isin(mask_array, (0, 1)).all():
            raise ValueError(f"La maschera non è binaria: {row['label_path']}")
        # Loss functions such as CrossEntropyLoss expect int64/long masks.
        mask = torch.from_numpy(mask_array.astype(np.int64, copy=False))

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask

    @staticmethod
    def _load_array(path: str | Path) -> np.ndarray:
        """Read a supported NumPy or TIFF file and return its values as an array."""
        path = Path(path)
        if path.suffix.lower() == ".npy":
            return np.load(path, allow_pickle=False)
        if path.suffix.lower() in {".tif", ".tiff"}:
            import tifffile

            return tifffile.imread(path)
        raise ValueError(f"Formato non supportato: {path}")


def create_datasets(
    train_fraction: float = 0.8,
    seed: int = 42,
    train_images_dir: str | Path = TRAIN_IMAGES_DIR,
    train_labels_dir: str | Path = TRAIN_LABELS_DIR,
    test_images_dir: str | Path = TEST_IMAGES_DIR,
    test_labels_dir: str | Path = TEST_LABELS_DIR,
) -> tuple[SWEDDataset, SWEDDataset, SWEDDataset]:
    """Create the three datasets used by the complete model workflow.

    Training and validation samples come from the training folders and are
    split by scene. Training samples receive random augmentation; validation
    and test samples do not. The returned order is training, validation, test.
    """
    development_pairs = get_pairs(train_images_dir, train_labels_dir)
    train_pairs, validation_pairs = split_pairs(development_pairs, train_fraction, seed)
    test_pairs = get_pairs(test_images_dir, test_labels_dir)

    train_dataset = SWEDDataset(train_pairs, transform=RandomGeometricTransform())
    validation_dataset = SWEDDataset(validation_pairs)
    test_dataset = SWEDDataset(test_pairs)
    return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    # This smoke test scans the filesystem only when the module runs as a script.
    train_dataset, validation_dataset, test_dataset = create_datasets()
    image, mask = train_dataset[0]
    print(f"Campioni: train={len(train_dataset)}, val={len(validation_dataset)}, test={len(test_dataset)}")
    print(f"Immagine: shape={tuple(image.shape)}, dtype={image.dtype}")
    print(f"Maschera: shape={tuple(mask.shape)}, dtype={mask.dtype}")
