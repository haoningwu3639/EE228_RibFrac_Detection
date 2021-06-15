
# EE228 Project - Rib Fracture Detection
Group assginment of EE228 Machine Learning, a Rib Fracture Detection algorithm based on 3D-UNet.

# Authors:
| Name | ID | E-mail |
| :-----: | :----: | :----: |
| Haoning Wu | 518030910285 | whn15698781666@sjtu.edu.cn |
| Longrun Zhi | 518030910320 | zlongrun@sjtu.edu.cn |

# Requirements
Code tested on following environments, other version should also work:
* python 3.7.9
* torch 1.4.0
* SimpleITK 1.2.4
* fastai 1.0.59
* matplotlib 3.1.3
* nibabel 3.0.0

# Project Structure
* [`checkpoint`](checkpoint): Checkpoint path. 
* [`data`](data): RibFrac Dataset, download at [Dataset](https://ribfrac.grand-challenge.org/dataset/) and place them in this folder.
* [`prediction_directory`](prediction_directory): Outputs of the model, compress it into a zip for submission.
* [`src`](src): Source code.
* [`src/train.py`](src/train.py): Train script.
    * CUDA_VISIBLE_DEVICES=0 python train.py --train_image_dir ... --train_label_dir ... --val_image_dir ... --val_label_dir ...
* [`src/test.py`](src/test.py): 
    * python --input_dir ... --output_dir ... --model_path ...

# Results 
 You can download our pretrained model and prediction results at [`pretrained_model`](pretrained_model) and [`results`](results). Our FROC can reach 0.5923 on test set and 0.3213 on validation set. We hope we can do better in the future. 

# Referneces
 If you gonna learn more about Rib Fracture Detection, we recommend these repositories for you:  [FracNet](https://github.com/M3DV/FracNet), [3DUNet-Pytorch](https://github.com/lee-zq/3DUNet-Pytorch), [3DUnetCNN](https://github.com/ellisdg/3DUnetCNN).

# Miscs
 Since we still have a lot of work to do, we have no time to maintain this project now. We hope that we can further improve this repository when we are free in the future. If you have any questions, please contact us by email. Thanks.