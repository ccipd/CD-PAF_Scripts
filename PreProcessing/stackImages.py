#This code stacks indivual image slices in png into a 3d image (mha). Does it for each patient

#The naming format for patients were case_XXXX_YYYY_0000 - to maintain nnUNet naming format

import os
import numpy as np
from PIL import Image
import SimpleITK as sitk

def load_and_stack_slices(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to hold slices for each case
    cases = {}

    # Gather all slices
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".png"):
            # Extract the case ID from the filename
            case_id = filename.split('_')[0:3]  # Get the first three parts
            case_id = '_'.join(case_id)          # Join to form case ID
            if case_id not in cases:
                cases[case_id] = []
            
            # Load the image slice
            img = Image.open(os.path.join(input_folder, filename))
            cases[case_id].append(np.array(img))
            print(f"Loaded: {filename} for case {case_id}")  # Debugging output

    # Process each case
    for case_id, slices in cases.items():
        # Check if slices list is empty
        if not slices:
            print(f"No slices found for case ID: {case_id}")
            continue

        # Stack slices along the third dimension to create a volume
        volume = np.stack(slices, axis=0)

      
        volume_sitk = sitk.GetImageFromArray(volume)

        # Save the volume as .mha file
        output_filename = f"{case_id}_Cropped.mha"
        sitk.WriteImage(volume_sitk, os.path.join(output_folder, output_filename))
        print(f"Saved {output_filename} to {output_folder}")

# Example usage
input_folder = 'Input Folder'  
output_folder = 'Output Folder'  
load_and_stack_slices(input_folder, output_folder)

