- Usage: `python train.py --dataset 38cloud --arch {ARCH} --img_ext .png --mask_ext .png --input_h 384 --input_w 384 --batch_size 4 --name 20240117v1_unetpp_vgg --train_dir {TRAIN_DIR} --eval_dir {EVAL_DIR}`  
    - `ARCH`: Choose among NestedUNet | v1 | v2. NestedUNet = v1 = vgg-based UNet, v2 = resnet50-based UNet  
    - `TRAIN_DIR`: Train root directory that contains `images` and `masks` directories  
    - `EVAL_DIR`: Evaluation root directory that contains `images` and `masks` directories  
        - Data architecture:
        ```
        <38 Cloud TRAIN/EVAL ROOT>
        ├── images
        |   ├── RGB_aaaa.png
        │   ├── RGB_bbbb.png
        │   ├── RGB_cccc.png
        │   ├── ...
        |
        └── masks
            ├── gt_aaaa.png
            ├── gt_bbbb.png
            ├── gt_cccc.png
            ├── ...
        ```


# PyTorch implementation of UNet++ (Nested U-Net)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

This repository contains code for a image segmentation model based on [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165) implemented in PyTorch.

[**NEW**] Add support for multi-class segmentation dataset.

[**NEW**] Add support for PyTorch 1.x.


## Requirements
- PyTorch 1.x or 0.41

## Installation
1. Create an anaconda environment.
```sh
conda create -n=<env_name> python=3.6 anaconda
conda activate <env_name>
```
2. Install PyTorch.
```sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
3. Install pip packages.
```sh
pip install -r requirements.txt
```

## Results
### DSB2018 (96x96)

Here is the results on DSB2018 dataset (96x96) with LovaszHingeLoss.

| Model                           |   IoU   |  Loss   |
|:------------------------------- |:-------:|:-------:|
| U-Net                           |  0.839  |  0.365  |
| Nested U-Net                    |  0.842  |**0.354**|
| Nested U-Net w/ Deepsupervision |**0.843**|  0.362  |
