import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage.measure import label, regionprops
from tqdm import tqdm

from dataset import TestDataset
from model import UNet3D, ResUNet, UNet
from post_processing import *
from transforms import *


def get_parser():
    
    parser = argparse.ArgumentParser(description='Input key parameters')

    parser.add_argument("--input_dir", type=str, default="../data/ribfrac-test-images/", help="The image nii directory.")
    parser.add_argument("--pred_dir", type=str, default="../prediction_directory/", help="The directory for saving predictions.")
    parser.add_argument("--model_path", default=None, help="The PyTorch model weight path.")
    parser.add_argument("--prob_thresh", default=0.3, help="Prediction probability threshold.")
    parser.add_argument("--bone_thresh", default=300, help="Bone binarization threshold.")
    parser.add_argument("--size_thresh", default=200, help="Prediction size threshold.")
    parser.add_argument("--batch_size", type=int, default=20, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4)

    return parser


def post_process(pred, image, prob_thresh, bone_thresh, size_thresh):
    # Post Processing  
    # remove non-rib predictions
    pred = remove_non_rib_pred(pred, image, 0.25)
    # remove connected regions with low confidence
    pred = remove_low_probs(pred, prob_thresh)
    # remove spine false positives
    pred = remove_spine_fp(pred, image, bone_thresh)
    # remove small connected regions
    pred = remove_small_things(pred, size_thresh)

    return pred

def predict(model, dataloader, prob_thresh, bone_thresh, size_thresh):
    pred = np.zeros(dataloader.dataset.image.shape)
    crop_size = dataloader.dataset.crop_size
    with torch.no_grad():
        for _, sample in enumerate(dataloader):
            images, centers = sample
            images = images.cuda()
            output = model(images).sigmoid().cpu().numpy().squeeze(axis=1)

            for i in range(len(centers)):
                center_x, center_y, center_z = centers[i]
                cur_pred_patch = pred[
                    center_x - crop_size // 2:center_x + crop_size // 2,
                    center_y - crop_size // 2:center_y + crop_size // 2,
                    center_z - crop_size // 2:center_z + crop_size // 2
                ]
                pred[
                    center_x - crop_size // 2:center_x + crop_size // 2,
                    center_y - crop_size // 2:center_y + crop_size // 2,
                    center_z - crop_size // 2:center_z + crop_size // 2
                ] = np.where(cur_pred_patch > 0, np.mean((output[i],
                    cur_pred_patch), axis=0), output[i])

    pred = post_process(pred, dataloader.dataset.image, prob_thresh, bone_thresh, size_thresh)

    return pred

def make_submission(pred, image_id, affine):
    # Make csv files for submission
    pred_label = label(pred > 0).astype(np.int16)
    pred_regions = regionprops(pred_label, pred)
    pred_index = [0] + [region.label for region in pred_regions]
    pred_proba = [0.0] + [region.mean_intensity for region in pred_regions]
    pred_label_code = [0] + [1] * int(pred_label.max())
    pred_image = nib.Nifti1Image(pred_label, affine)
    pred_info = pd.DataFrame({
        "public_id": [image_id] * len(pred_index),
        "label_id": pred_index,
        "confidence": pred_proba,
        "label_code": pred_label_code
    })

    return pred_image, pred_info


def test():
    # Add command line arguments
    parser = get_parser()
    args = parser.parse_args()

    model = UNet3D(1, 1, first_out_channels=16)
    model.eval()
    # Load model to predict
    if args.model_path is not None:
        model_weights = torch.load(args.model_path)
        model.load_state_dict(model_weights)
    model = nn.DataParallel(model).cuda()

    transforms = [
        Window(-100, 1000),
        # Resample(),
        # Equalize(),
        Normalize(-100, 1000)
    ]

    image_path_list = sorted([os.path.join(args.input_dir, file) for file in os.listdir(args.input_dir) if "nii" in file])
    image_id_list = [os.path.basename(path).split("-")[0] for path in image_path_list]

    progress = tqdm(total=len(image_id_list))
    pred_info_list = []
    # Prediction
    for image_id, image_path in zip(image_id_list, image_path_list):
        dataset = TestDataset(image_path, transforms=transforms)
        dataloader = TestDataset.get_dataloader(dataset, args.batch_size, args.num_workers)
        pred_arr = predict(model, dataloader, args.prob_thresh, args.bone_thresh, args.size_thresh)
        pred_image, pred_info = make_submission(pred_arr, image_id, dataset.image_affine)
        pred_info_list.append(pred_info)
        pred_path = os.path.join(args.output_dir, f"{image_id}-label.nii.gz")
        nib.save(pred_image, pred_path)

        progress.update()

    pred_info = pd.concat(pred_info_list, ignore_index=True)
    pred_info.to_csv(os.path.join(args.output_dir, "ribfrac-test-pred.csv"), index=False)


if __name__ == "__main__":
    test()
