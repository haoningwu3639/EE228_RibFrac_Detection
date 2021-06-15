import argparse
from functools import partial

import torch
import torch.nn as nn
from fastai.basic_train import Learner
from fastai.train import ShowGraph
from fastai.callbacks.tracker import SaveModelCallback
from fastai.data_block import DataBunch
from torch import optim

from transforms import *
from dataset import TrainDataset
from losses import DiceLoss, MixLoss, SoftDiceLoss
from metrics import dice, fbeta_score, precision, recall
from model import UNet3D, ResUNet, UNet


def get_parser():
    
    parser = argparse.ArgumentParser(description='Input train image and validation image paths and some key parameters')
    
    parser.add_argument("--train_image_dir", type=str, default="../data/ribfrac-train-images/", help="The training image nii directory.")
    parser.add_argument("--train_label_dir", type=str, default="../data/ribfrac-train-labels/", help="The training label nii directory.")
    parser.add_argument("--val_image_dir", type=str, default="../data/ribfrac-val-images/", help="The validation image nii directory.")
    parser.add_argument("--val_label_dir", type=str, default="../data/ribfrac-val-labels/", help="The validation label nii directory.")
    parser.add_argument("--batch_size", type=int, default=20, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--load_model", type=str, default="", help="Load old model to finetune")
    parser.add_argument("--save_path", type=str, default="../checkpoint/model.pth")

    return parser

def train():
    # Add command line arguments
    parser = get_parser()
    args = parser.parse_args()

    train_image_dir = args.train_image_dir
    train_label_dir = args.train_label_dir
    val_image_dir = args.val_image_dir
    val_label_dir = args.val_label_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    lr = args.lr
    save_path = args.save_path
    load_model = args.load_model

    optimizer = optim.Adam
    criterion = MixLoss(nn.BCEWithLogitsLoss(), 0.5, DiceLoss(), 1)

    thresh = 0.1
    recall_partial = partial(recall, thresh=thresh)
    precision_partial = partial(precision, thresh=thresh)
    fbeta_score_partial = partial(fbeta_score, thresh=thresh)

    model = UNet3D(1, 1, first_out_channels=16)
    # Load model to finetune
    if load_model:
        model.load_state_dict(torch.load(load_model))
    model = nn.DataParallel(model.cuda())

    transforms = [
        Window(-100, 1000),
        # Resample(),
        # Equalize(),
        Normalize(-100, 1000)
    ]

    train_dataset = TrainDataset(train_image_dir, train_label_dir, transforms=transforms)
    train_dataloader = TrainDataset.get_dataloader(train_dataset, batch_size, False, num_workers)
    val_dataset = TrainDataset(val_image_dir, val_label_dir, transforms=transforms)
    val_dataloader = TrainDataset.get_dataloader(val_dataset, batch_size, False, num_workers)

    databunch = DataBunch(train_dataloader, val_dataloader, collate_fn=TrainDataset.collate_fn)
    # Save model each epoch
    learn = Learner(databunch, model, opt_func=optimizer, loss_func=criterion, metrics=[dice, recall_partial, precision_partial, fbeta_score_partial])
    learn.fit_one_cycle(epochs, lr, pct_start=0, div_factor=1000, callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy', name='best')])
    # Save the final model
    torch.save(model.module.state_dict(), save_path)

if __name__ == '__main__':
    train()
