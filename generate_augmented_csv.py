import pandas as pd
import numpy as np
from pathlib import Path
import SimpleITK as sitk

# Your parameters & paths
PATCH_SIZE = np.array([128, 128, 64])
AUGMENTATIONS = ['orig', 'flip_x', 'flip_y', 'flip_z', 'rot90_xy']

csv_path = Path("C:/Users/prime/Desktop/luna25-baseline-public/csv/LUNA25_Public_Training_Development_Data.csv")
image_root = Path("C:/Users/prime/Desktop/Lung Cancer/luna25-baseline-public-main/data/luna25_images/luna25_images")
output_csv_path = Path("C:/Users/prime/Desktop/luna25-baseline-public/data/data_augmented/LUNA25_Public_Training_Development_Data_augmented.csv")

df = pd.read_csv(csv_path)

# You may want to keep all columns or a subset
columns_to_keep = ["PatientID", "SeriesInstanceUID", "StudyDate", "CoordX", "CoordY", "CoordZ", 
                   "LesionID", "AnnotationID", "NoduleID", "label", "Age_at_StudyDate", "Gender"]

# Filter to only columns present in df
columns_to_keep = [c for c in columns_to_keep if c in df.columns]

augmented_rows = []

for series_uid, group in df.groupby("SeriesInstanceUID"):
    mha_file = image_root / f"{series_uid}.mha"
    if not mha_file.exists():
        print(f"Missing image for {series_uid}, skipping.")
        continue

    # Read the image and metadata for coordinate transformations
    image = sitk.ReadImage(str(mha_file))
    spacing = np.array(image.GetSpacing())[::-1]
    origin = np.array(image.GetOrigin())[::-1]
    size = np.array(image.GetSize())[::-1]
    
    extent = PATCH_SIZE // 2

    for _, row in group.iterrows():
        # Original physical coordinates

        coord_world = np.array([row["CoordX"], row["CoordY"], row["CoordZ"]])

        # Convert world coordinate to index (voxel coordinate)
        coord_index = np.array(image.TransformPhysicalPointToIndex(coord_world.tolist()))
        
        # Pad if needed (optional, like in your original script)
        need_pad = np.any(coord_index - extent < 0) or np.any(coord_index + extent > size)
        if need_pad:
            # Pad image if necessary (same as your augmentation script)
            image = sitk.ConstantPad(image, extent.tolist(), extent.tolist(), -1024)
            coord_index = np.array(image.TransformPhysicalPointToIndex(coord_world.tolist()))
            size = np.array(image.GetSize())[::-1]

        # Now apply transformations to coordinates for each augmentation:

        # Define a helper function to get the new index coords after augmentation
        def transform_coord(idx, aug):
            x, y, z = idx
            sx, sy, sz = size
            if aug == 'orig':
                return np.array([x, y, z])
            elif aug == 'flip_x':
                # flip along x axis: x -> sx - 1 - x
                return np.array([sx - 1 - x, y, z])
            elif aug == 'flip_y':
                # flip along y axis: y -> sy - 1 - y
                return np.array([x, sy - 1 - y, z])
            elif aug == 'flip_z':
                # flip along z axis: z -> sz - 1 - z
                return np.array([x, y, sz - 1 - z])
            elif aug == 'rot90_xy':
                # rotate 90 degrees CCW in xy plane: (x, y) -> (y, sx - 1 - x)
                return np.array([y, sx - 1 - x, z])
            else:
                return idx

        for aug in AUGMENTATIONS:
            new_idx = transform_coord(coord_index, aug)
            
            # Convert back to physical coords (world)
            new_phys = image.TransformIndexToPhysicalPoint(new_idx.tolist())
            new_phys = np.array(new_phys)

            # Create a new row with updated coordinates
            new_row = row.copy()
            new_row["CoordX"] = new_phys[0]
            new_row["CoordY"] = new_phys[1]
            new_row["CoordZ"] = new_phys[2]
            
            # Optionally, you can update NoduleID or AnnotationID to indicate augmentation if you want
            # e.g., new_row["NoduleID"] = f"{row['NoduleID']}_{aug}"
            # But if you want to keep them the same, leave as is
            augmented_rows.append(new_row[columns_to_keep])

# Create new dataframe and save CSV
df_augmented = pd.DataFrame(augmented_rows)
df_augmented.to_csv(output_csv_path, index=False)

print(f"Augmented CSV saved to {output_csv_path}")
