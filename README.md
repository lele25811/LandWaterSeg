# LandWaterSeg

University project for binary semantic segmentation of land and water in Sentinel-2 satellite imagery. The model is a U-Net with a MobileNetV2 encoder, trained from scratch on six spectral bands from the Sentinel-2 Water Dataset (SWED).

## Repository contents

| File | Purpose |
| --- | --- |
| `Semantic Segmentation LandWater.pdf` | Project presentation introducing the land/water semantic segmentation project. |
| `datasetsetup.py` | Pair images and masks, validate labels, create scene-disjoint splits, normalize spectral bands, and apply training augmentation. |
| `train_unet_mobilenet_v2_6bands.py` | Define and train the six-band U-Net, save the best checkpoint, and evaluate it on the test split. |
| `test_unet_mobilenet_v2_6bands.ipynb` | Load a checkpoint, compute test metrics, and display satellite images alongside reference and predicted masks. |
| `viewdataset.ipynb` | Inspect SWED NumPy samples, masks, and overlays using the first three image bands for display. |
| `pyproject.toml` | Python version requirement and project dependencies. |

The dataset, trained weights are not included in the committed files.

## Dataset preparation

Obtain SWED from the [UK Hydrographic Office open data portal](https://openmldata.ukho.gov.uk/) and extract it into this layout:

```text
LandWaterSeg/
├── datasetsetup.py
├── train_unet_mobilenet_v2_6bands.py
└── Sentinel-2WaterDataset(SWED)/
    └── SWED/
        └── train/
            ├── images/
            │   └── <scene>_image_<row>_<column>.npy
            └── labels/
                └── <scene>_chip_<row>_<column>.npy
```

`datasetsetup.py` supports `.npy`, `.tif`, and `.tiff` files. Images are paired with masks by their normalized filename identifiers; masks can use either `_chip_` or `_label_`. Missing counterparts and duplicate identifiers raise errors. The visualization notebook handles `.npy` pairs with the `_image_`/`_chip_` naming convention.

Each image must contain 12 bands in HWC or CHW order, with a spatial size of **256 × 256**. Masks must reduce to **256 × 256** after removing singleton dimensions and contain only these class values:

| Value | Class |
| --- | --- |
| `0` | Land |
| `1` | Water |

Masks with invalid shapes or nonbinary values are excluded before splitting.

### Spectral preprocessing

The loader assumes the source band order is `B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12`. It selects:

| Model channel | Sentinel-2 band | Description |
| --- | --- | --- |
| 1 | B02 | Blue |
| 2 | B03 | Green |
| 3 | B04 | Red |
| 4 | B08 | Near infrared |
| 5 | B11 | Short-wave infrared 1 |
| 6 | B12 | Short-wave infrared 2 |

Values are divided by **8160.0** and clipped to `[0, 1]`. The dataset returns `float32` image tensors shaped `[6, 256, 256]` and `int64` masks shaped `[256, 256]`.

### Splits and augmentation

All three subsets are derived from **SWED/train**, with target proportions of **50% training, 30% validation, and 20% testing**. Complete scenes are assigned to a single subset to prevent leakage between tiles from the same scene. The split balances sample counts and water-pixel prevalence; actual proportions can differ because scenes remain intact. Dataset creation defaults to seed `42`.

Training augmentation applies synchronized horizontal and vertical flips, a random rotation of 0°, 90°, 180°, or 270°, and image-only Gaussian noise with probability `0.3` and standard deviation `0.005`. Validation and test samples are not augmented.

Check dataset preparation with:

```bash
python datasetsetup.py
```

This prints split sizes and the shape and type of a training image and mask. Open `viewdataset.ipynb` to inspect samples visually.

## Model and training

The model uses `segmentation_models_pytorch.Unet` with a MobileNetV2 encoder and **no pretrained weights**. It produces one water logit per pixel and trains with binary cross-entropy with logits. Its wrapper returns two probability channels ordered as `[P(land), P(water)]`, plus the loss when labels are supplied.

Default settings are defined as constants in `train_unet_mobilenet_v2_6bands.py`:

| Setting | Value |
| --- | --- |
| Encoder depth | 4 |
| Decoder channels | `(128, 64, 32, 16)` |
| Maximum epochs | 50 |
| Batch size | 8 |
| Gradient accumulation | 2 batches; effective batch size 16 |
| Optimizer | AdamW |
| Learning rate | `5e-5` |
| Weight decay | `1e-4` |
| Learning-rate scheduler | ReduceLROnPlateau on validation IoU; factor `0.5`, patience `2` |
| Early stopping | 10 epochs without validation IoU improvement |

## Evaluation and visualization

After producing or supplying a compatible checkpoint and restoring the missing shared utilities, open `test_unet_mobilenet_v2_6bands.ipynb` from the repository root and run its cells in order.

The notebook validates checkpoint metadata, recreates the test split, and reports accuracy, water IoU, precision, recall, and Dice. Metrics aggregate pixel counts over the full test set; water is the positive class. Predictions use `argmax` over the two probability channels.

It also displays random test samples with an RGB composite, ground truth, and prediction. RGB uses B04/B03/B02 with a display-only percentile stretch; inference uses all six normalized bands. Masks show land in black and water in blue. Change `N_IMAGES` to adjust the number of displayed samples.

To reproduce a training run's test split, retain the same dataset files, split settings, and training seed: the notebook recreates the split rather than loading a saved split manifest.

