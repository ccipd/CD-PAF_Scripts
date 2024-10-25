"""""
The following code generates one large bounding box given that there are 3 regions of interests 
The bounding boxes are generated using the ground truth label
""""

import os
import SimpleITK as sitk
import numpy as np
from skimage.measure import regionprops, label

def generate_bounding_boxes(image_folder, label_folder, output_folder):
    """
    Generate bounding boxes for three regions per slice in 3D MHA images.
    
    :param image_folder: Path to the folder containing input 3D MHA image files
    :param label_folder: Path to the folder containing input 3D MHA label files
    :param output_folder: Path to the folder to save output 3D MHA images with bounding boxes
    """
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.mha')]
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.mha')]
    
    for image_file in image_files:
        image_base = '_'.join(image_file.split('_')[:-1])
        label_file = next((l for l in label_files if l.startswith(image_base)), None)
        
        if label_file is None:
            print(f"No corresponding label found for {image_file}. Skipping.")
            continue
        
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, label_file)
        output_path = os.path.join(output_folder, f"bbox_{image_file}")
        
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(label_path)
        
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        
        print(f"Image shape: {image_array.shape}, Number of channels: {image_array.shape[-1] if len(image_array.shape) > 3 else 1}")
        print(f"Mask shape: {mask_array.shape}, Number of channels: {mask_array.shape[-1] if len(mask_array.shape) > 3 else 1}")
        
        bbox_image = image_array.copy()
        
        for z in range(image_array.shape[0]):  
            if z >= mask_array.shape[0]:
                break
            slice_mask = mask_array[z]
            labeled_mask = label(slice_mask)
            regions = regionprops(labeled_mask)
            
            if not regions:
                continue  # Skip this slice if no regions found
            
            # Find the largest bounding box for this slice
            min_r, min_c = float('inf'), float('inf')
            max_r, max_c = float('-inf'), float('-inf')
            
            for region in regions:
                minr, minc, maxr, maxc = region.bbox
                min_r = min(min_r, minr)
                min_c = min(min_c, minc)
                max_r = max(max_r, maxr)
                max_c = max(max_c, maxc)
            
            # Draw the largest bounding box for this slice
            bbox_image[z, min_r, min_c:max_c] = np.max(image_array)  # Top edge
            bbox_image[z, max_r-1, min_c:max_c] = np.max(image_array)  # Bottom edge
            bbox_image[z, min_r:max_r, min_c] = np.max(image_array)  # Left edge
            bbox_image[z, min_r:max_r, max_c-1] = np.max(image_array)  # Right edge
        
        bbox_sitk = sitk.GetImageFromArray(bbox_image)
        bbox_sitk.CopyInformation(image)
        
        sitk.WriteImage(bbox_sitk, output_path)
        
        print(f"Processed {image_file}")


def print_image_info(image_folder, label_folder):
    """
    Print information about the number of channels in the images and labels.
    
    :param image_folder: Path to the folder containing input 3D MHA image files
    :param label_folder: Path to the folder containing input 3D MHA label files
    """
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.mha')]
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.mha')]
    
    for image_file in image_files:
        image_base = '_'.join(image_file.split('_')[:-1])
        label_file = next((l for l in label_files if l.startswith(image_base)), None)
        
        if label_file is None:
            print(f"No corresponding label found for {image_file}. Skipping.")
            continue
        
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, label_file)
        
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(label_path)
        
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        
        print(f"File: {image_file}")
        print(f"Image shape: {image_array.shape}, Number of channels: {image_array.shape[-1] if len(image_array.shape) > 3 else 1}")
        print(f"Mask shape: {mask_array.shape}, Number of channels: {mask_array.shape[-1] if len(mask_array.shape) > 3 else 1}")
        print()

# Example usage:
#generate_bounding_boxes("Image File Path", "Label_File Path", "Output_File Path")
#print_image_info("Image_File Path", "Label_File Path")
