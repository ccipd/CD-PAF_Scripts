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


MedSAM_CKPT_PATH = "/mnt/vstor/CSE_CSDS_VXC204/mxh1029/notebooks/medsam_vit_b.pth"
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

    num_features, bboxes =  bbox_per_mask(mask2d)

    if num_features >0:
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

        medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)

        if len(medsam_seg.shape) > 2:
            gen_mask = np.sum(medsam_seg, axis=0)
        else:
            gen_mask = medsam_seg

        gen_mask = np.where(gen_mask > 0, 1, 0)

        return gen_mask
    
    else:
        return None

for idx, image_path in enumerate(image_path_base.glob("*.mha")):
    mask_path = Path('_'.join(image_path.stem.split("_")[:-1])).with_suffix(".mha")
    mask_path = mask_path_base / mask_path

    image = sitk.ReadImage(str(image_path))
    mask_orig = sitk.ReadImage(str(mask_path))

    image = sitk.GetArrayFromImage(image)
    mask = sitk.GetArrayFromImage(mask_orig)

    mask1 = mask.copy()
    mask2 = mask.copy()
    mask3 = mask.copy()

    mask1[mask1 == 2] = 0
    mask1[mask1 == 3] = 0


    mask2[mask2 == 1] = 0
    mask2[mask2 == 3] = 0

    mask3[mask3 == 1] = 0
    mask3[mask3 == 2] = 0

    num_slices = image.shape[0]

    gen_mask1_3d = np.zeros(mask1.shape)
    gen_mask2_3d = np.zeros(mask2.shape)
    gen_mask3_3d = np.zeros(mask3.shape)    

    dice_dict = {"slice": [], "mask1": [], "mask2": [], "mask3": []}

    for slice in range(num_slices):

        
        print("Index: ", idx, "Slice: ", slice)

        dice1 = None
        dice2 = None
        dice3 = None

        im2d = image[slice,:,:]
        mask1_2d = mask1[slice,:,:]
        mask2_2d = mask2[slice,:,:]
        mask3_2d = mask3[slice,:,:]

        img_1024_tensor, H, W = preprocess_image(im2d)

        gen_mask1_2d = run(mask1_2d, img_1024_tensor, H, W)
        gen_mask2_2d = run(mask2_2d, img_1024_tensor, H, W)
        gen_mask3_2d = run(mask3_2d, img_1024_tensor, H, W)

        # print("Gen Mask 1: ", gen_mask1_2d.shape)
        # print("Gen Mask 2: ", gen_mask2_2d.shape)
        # print("Gen Mask 3: ", gen_mask3_2d.shape)

        # print("Mask 1: ", mask1_2d.shape)
        # print("Mask 2: ", mask2_2d.shape)
        # print("Mask 3: ", mask3_2d.shape)
        if gen_mask1_2d is not None:
            unique_values = np.unique(gen_mask1_2d)
            dice1 = None

            if len(unique_values) == 2:
                gen_mask1_3d[slice, :, :] = gen_mask1_2d
                dice1 = dice_fn(mask1_2d, gen_mask1_2d)
                if dice1 < 0.1:
                    dice1 = None
        else:
            dice1 = None

        if gen_mask2_2d is not None:
            unique_values = np.unique(gen_mask2_2d)
            dice2 = None

            if len(unique_values) == 2:
                gen_mask2_3d[slice, :, :] = gen_mask2_2d
                dice2 = dice_fn(mask2_2d, gen_mask2_2d)
                if dice2 < 0.1:
                    dice2 = None
        else:
            dice2 = None

        if gen_mask3_2d is not None:
            unique_values = np.unique(gen_mask3_2d)
            dice3 = None

            if len(unique_values) == 2:
                gen_mask3_3d[slice, :, :] = gen_mask3_2d
                dice3 = dice_fn(mask3_2d, gen_mask3_2d)
                if dice3 < 0.1:
                    dice3 = None
        else:
            dice3 = None


        dice_dict["slice"].append(slice)
        dice_dict["mask1"].append(dice1)
        dice_dict["mask2"].append(dice2)
        dice_dict["mask3"].append(dice3)


    gen_mask1 = np.where(gen_mask1_3d ==0, 0, 1)
    gen_mask2 = np.where(gen_mask2_3d ==0, 0, 2)
    gen_mask3 = np.where(gen_mask3_3d ==0, 0, 3)



    gen_maks_1 = sitk.GetImageFromArray(gen_mask1)
    gen_maks_1.CopyInformation(mask_orig)

    gen_maks_2 = sitk.GetImageFromArray(gen_mask2)
    gen_maks_2.CopyInformation(mask_orig)

    gen_maks_3 = sitk.GetImageFromArray(gen_mask3)
    gen_maks_3.CopyInformation(mask_orig)

    sitk.WriteImage(gen_maks_1, f"./gen/gen_mask1_{image_path.stem}.mha")
    sitk.WriteImage(gen_maks_2, f"./gen/gen_mask2_{image_path.stem}.mha")
    sitk.WriteImage(gen_maks_3, f"./gen/gen_mask3_{image_path.stem}.mha")

    df = pd.DataFrame(dice_dict)
    df.to_csv(f"./gen/dice_{image_path.stem}.csv", index=False)

    # gen_mask = gen_mask1 + gen_mask2 + gen_mask3

    # gen_mask = sitk.GetImageFromArray(gen_mask)
    # gen_mask.CopyInformation(mask_orig)

    # sitk.WriteImage(gen_mask, f"./gen/gen_mask_{image_path.stem}.mha")

