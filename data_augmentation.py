from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import random

# Parameters
PATCH_SIZE = np.array([128, 128, 64])
AUGMENTATIONS = ['flip_x', 'flip_y', 'flip_z', 'rot90_xy']
SAVE_FORMAT = ".npy"

# Input paths
csv_path = Path("C:/Users/prime/Desktop/luna25-baseline-public/csv/LUNA25_Public_Training_Development_Data.csv")  # Replace with actual CSV
image_root = Path("C:/Users/prime/Desktop/Lung Cancer/luna25-baseline-public-main/data/luna25_images/luna25_images")     # MHA directory
output_image_dir = Path("C:/Users/prime/Desktop/luna25-baseline-public/data/data_augmented/image")  # To save .npy image blocks
output_metadata_dir = Path("C:/Users/prime/Desktop/luna25-baseline-public/data/data_augmented/metadata")  # To save metadata

# Ensure output directories exist
output_image_dir.mkdir(parents=True, exist_ok=True)
output_metadata_dir.mkdir(parents=True, exist_ok=True)

# Load CSV and clean
df = pd.read_csv(csv_path)
required_cols = ["PatientID", "SeriesInstanceUID", "StudyDate", "CoordX", "CoordY", "CoordZ", "NoduleID"]
df = df[[col for col in required_cols if col in df.columns]]

# Group by SeriesInstanceUID to avoid loading image repeatedly
for series_uid, group in df.groupby("SeriesInstanceUID"):
    mha_file = image_root / f"{series_uid}.mha"
    if not mha_file.exists():
        print(series_uid)
        continue

    # Read the image
    image = sitk.ReadImage(str(mha_file))
    spacing = np.array(image.GetSpacing())[::-1]
    origin = np.array(image.GetOrigin())[::-1]
    transform = np.array(image.GetDirection()).reshape(3, 3)

    extent = PATCH_SIZE // 2

    for _, row in group.iterrows():
        coord_world = np.array([row["CoordX"], row["CoordY"], row["CoordZ"]])
        coord_index = image.TransformPhysicalPointToIndex(coord_world.tolist())

        # Padding if necessary
        size = np.array(image.GetSize())[::-1]
        need_pad = np.any(coord_index - extent < 0) or np.any(coord_index + extent > size)
        if need_pad:
            image = sitk.ConstantPad(image, extent.tolist(), extent.tolist(), -1024)
            coord_index = np.array(image.TransformPhysicalPointToIndex(coord_world.tolist()))

        # Extract original patch
        patch = image[
            int(coord_index[0] - extent[0]):int(coord_index[0] + extent[0]),
            int(coord_index[1] - extent[1]):int(coord_index[1] + extent[1]),
            int(coord_index[2] - extent[2]):int(coord_index[2] + extent[2])
        ]
        patch_array = sitk.GetArrayFromImage(patch)

        # Apply and save each augmentation
        for aug in AUGMENTATIONS:
            aug_patch = patch_array.copy()
            if aug == 'flip_x':
                aug_patch = np.flip(aug_patch, axis=2)
            elif aug == 'flip_y':
                aug_patch = np.flip(aug_patch, axis=1)
            elif aug == 'flip_z':
                aug_patch = np.flip(aug_patch, axis=0)
            elif aug == 'rot90_xy':
                aug_patch = np.rot90(aug_patch, k=1, axes=(1, 2))

            out_name = f"{row['NoduleID']}_{int(row['StudyDate'])}_{aug}"
            np.save(output_image_dir / f"{out_name}.npy", aug_patch)

            metadata = {
                'origin': origin,
                'spacing': spacing,
                'transform': np.eye(3),
            }
            np.save(output_metadata_dir / f"{out_name}.npy", np.array([metadata], dtype=object))
