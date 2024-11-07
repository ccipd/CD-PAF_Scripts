import torch
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
from segment_anything import sam_model_registry
from skimage import transform
import torch.nn.functional as F
from scipy.ndimage import label, find_objects
from segment_anything import sam_model_registry

import pandas as pd 


MedSAM_CKPT_PATH = "/mnt/pan/Data7/axs2220/MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
device = "cuda:0"
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
medsam_model = medsam_model.to(device)
medsam_model.eval()

image_path_base = Path("/mnt/pan/Data7/axs2220/nnUNet/nnUNetFrame/dataset/nnUNet_raw/nnUNet_raw_data/Dataset001_Fistula/imagesTr")
mask_path_base = Path("/mnt/pan/Data7/axs2220/nnUNet/nnUNetFrame/dataset/nnUNet_raw/nnUNet_raw_data/Dataset001_Fistula/labelsTr") 


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )


    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )
    


    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def dice_fn(a,b):
    a = a.to(bool) if isinstance(a, torch.Tensor) else a.astype(bool)
    b = b.to(bool) if isinstance(b, torch.Tensor) else b.astype(bool)
    if a.sum() == 0 and b.sum() == 0: raise ValueError
    return 2 * (a & b).sum() / (a.sum() + b.sum())

def bbox_per_mask(mask2d):

    labeled_mask, num_features = label(mask2d)
    bounding_boxes = find_objects(labeled_mask)


    return num_features, bounding_boxes

def preprocess_image(im2d):

    img_np = im2d
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape

    img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)

    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)  # normalize to [0, 1], (H, W, 3)

    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    return img_1024_tensor, H, W


def run(mask2d, img_1024_tensor, H, W):
    num_features, bboxes = bbox_per_mask(mask2d)
    print(f"Number of features found: {num_features}")

    if num_features > 0:
        all_boxes = []

        for bbox in bboxes:
            y_min, y_max = bbox[0].start, bbox[0].stop
            x_min, x_max = bbox[1].start, bbox[1].stop
            box_np = np.array([[x_min, y_min, x_max, y_max]])
            all_boxes.append(box_np)
        
        all_boxes = np.concatenate(all_boxes, axis=0)
        box_1024 = all_boxes / np.array([W, H, W, H]) * 1024

        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)

        medsam_prob = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
        medsam_seg = (medsam_prob > 0.5).astype(np.uint8)

        # Ensure output is 2D
        return np.squeeze(medsam_seg), np.squeeze(medsam_prob)
    
    else:
        # Return tuple of None instead of just None
        return None, None

for idx, image_path in enumerate(image_path_base.glob("*.mha")):
    print(f"Processing image: {image_path.stem}")
    
    mask_path = Path('_'.join(image_path.stem.split("_")[:-1])).with_suffix(".mha")
    mask_path = mask_path_base / mask_path

    image = sitk.ReadImage(str(image_path))
    mask_orig = sitk.ReadImage(str(mask_path))

    # Convert to numpy arrays
    image = sitk.GetArrayFromImage(image)
    mask = sitk.GetArrayFromImage(mask_orig)
    
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")

    # Initialize probability maps for each class
    prob_maps = np.zeros((4,) + image.shape, dtype=np.float32)  # 4 channels: background + 3 classes
    
    for slice_idx in range(image.shape[0]):
        print(f"Processing slice {slice_idx}/{image.shape[0]}")
        
        # Get 2D slice and ensure it's 2D
        im2d = image[slice_idx,:,:]
        print(f"Slice shape: {im2d.shape}")
        
        img_1024_tensor, H, W = preprocess_image(im2d)
        print(f"Preprocessed tensor shape: {img_1024_tensor.shape}")
        
        # Get image embedding once per slice
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)
        
        # Process each class
        for class_idx in range(1, 4):  # Classes 1, 2, 3
            mask_2d = (mask[slice_idx,:,:] == class_idx).astype(np.uint8)
            print(f"Processing class {class_idx}, mask shape: {mask_2d.shape}")
            
            # Get bounding boxes for this class
            num_features, bboxes = bbox_per_mask(mask_2d)
            
            if num_features > 0:
                # Process bounding boxes
                all_boxes = []
                for bbox in bboxes:
                    y_min, y_max = bbox[0].start, bbox[0].stop
                    x_min, x_max = bbox[1].start, bbox[1].stop
                    box_np = np.array([[x_min, y_min, x_max, y_max]])
                    all_boxes.append(box_np)
                
                all_boxes = np.concatenate(all_boxes, axis=0)
                box_1024 = all_boxes / np.array([W, H, W, H]) * 1024
                
                # Get probability map for this class and ensure it's 2D
                prob_map = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
                prob_map = np.squeeze(prob_map)  # Remove extra dimensions
                
                print(f"Probability map shape before assignment: {prob_map.shape}")
                
                if prob_map.ndim > 2:
                    print(f"Warning: prob_map has extra dimensions: {prob_map.shape}")
                    prob_map = prob_map[0] if prob_map.shape[0] == 1 else prob_map[-1]  # Take first or last slice
                
                # Store in the probability maps array
                prob_maps[class_idx, slice_idx, :, :] = prob_map
            else:
                # If no features found, set probabilities to 0
                prob_maps[class_idx, slice_idx, :, :] = 0.0
    
    # Calculate background probability (1 - max of other probabilities)
    prob_maps[0] = 1.0 - np.max(prob_maps[1:], axis=0)
    
    # Save probability maps
    output_path = Path("/mnt/pan/Data7/axs2220/MedSAM/output/prob_maps")
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Saving probability maps...")
    
    # Save each class separately
    for class_idx in range(4):
        # Transpose the array to match SimpleITK's expected ordering
        prob_map = prob_maps[class_idx]
        prob_map_sitk = sitk.GetImageFromArray(prob_map)
        
        # Create a new image with the same metadata as original
        prob_map_sitk.SetSpacing(mask_orig.GetSpacing())
        prob_map_sitk.SetDirection(mask_orig.GetDirection())
        prob_map_sitk.SetOrigin(mask_orig.GetOrigin())
        
        output_name = f"{image_path.stem}_prob_class{class_idx}.mha"
        output_file = output_path / output_name
        sitk.WriteImage(prob_map_sitk, str(output_file))
        print(f"Saved {output_file}")
    
    # Save combined probability maps
    # Transpose the array to match SimpleITK's expected ordering (channels last)
    prob_maps_combined = np.moveaxis(prob_maps, 0, -1)  # Move channels to last axis
    prob_maps_combined = sitk.GetImageFromArray(prob_maps_combined)
    
    # Set metadata manually instead of copying
    prob_maps_combined.SetSpacing(mask_orig.GetSpacing())
    prob_maps_combined.SetDirection(mask_orig.GetDirection())
    prob_maps_combined.SetOrigin(mask_orig.GetOrigin())
    
    output_file = output_path / f"{image_path.stem}_prob_maps_all.mha"
    sitk.WriteImage(prob_maps_combined, str(output_file))
    print(f"Saved combined probability maps to {output_file}")



