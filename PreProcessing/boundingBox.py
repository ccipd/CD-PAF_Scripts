from pathlib import Path
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import label, find_objects
import os

# import libraries
import pandas as pd

import torch
from scipy.ndimage import label, center_of_mass, find_objects


# Path
image_path_base = Path("Enter Image path")
mask_path_base = Path("Enter Labels Path")

# Mohsen + Pytoch + Claude
def add_bbox(mask2d, im2d, output_path):
    """
    Add bounding boxes to an image based on a mask and save the result.

    Args:
        mask2d (np.array): A 2D binary mask which includes all the areas.
        im2d (np.array): A 2D image.
        output_path (str): Path to save the output image.

    Returns:
        int: The number of areas.
        list: The bounding boxes of the areas.
    """
    if isinstance(mask2d, torch.Tensor):
        mask2d = mask2d.numpy()

    labeled_mask, num_features = label(mask2d, structure=np.ones((3, 3)))
    bounding_boxes = find_objects(labeled_mask)

    if num_features > 0:
        plt.figure(figsize=(10, 10))
        plt.imshow(im2d, cmap='gray')
        plt.contour(mask2d, levels=[0.5], colors='g')

        for bbox in bounding_boxes:
            y_min, y_max = bbox[0].start, bbox[0].stop
            x_min, x_max = bbox[1].start, bbox[1].stop
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                              edgecolor='red', facecolor='none', linewidth=2))
        
        plt.savefig(output_path)
        plt.close()

    return num_features, bounding_boxes

# Output
output_folder = Path("Output Folder Path")
output_folder.mkdir(exist_ok=True)

# For all Images
for image_path in image_path_base.glob("*.mha"):
    mask_path = Path('_'.join(image_path.stem.split("_")[:-1])).with_suffix(".mha")
    mask_path = mask_path_base / mask_path

    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    image = sitk.GetArrayFromImage(image)
    mask = sitk.GetArrayFromImage(mask)

    # Process each slice of the 3D image
    for slice_index in range(image.shape[0]):
        mask_slice = mask[slice_index]
        
        # Check if the slice has any mask
        if np.any(mask_slice):
            mask1 = mask_slice.copy()
            mask2 = mask_slice.copy()
            mask3 = mask_slice.copy()

            mask1[mask1 == 2] = 0
            mask1[mask1 == 3] = 0

            mask2[mask2 == 1] = 0
            mask2[mask2 == 3] = 0

            mask3[mask3 == 1] = 0
            mask3[mask3 == 2] = 0

            # Generate and save bounding boxes for each mask
            for mask_index, mask_slice in enumerate([mask1, mask2, mask3]):
                if np.any(mask_slice):
                    output_path = output_folder / f"{image_path.stem}_slice{slice_index}_mask{mask_index+1}.png"
                    num_features, _ = add_bbox(mask_slice, image[slice_index], str(output_path))
                    if num_features > 0:
                        print(f"Processed: {image_path.name}, Slice: {slice_index}, Mask: {mask_index+1}")

print("All images processed. Results saved in:", output_folder)
