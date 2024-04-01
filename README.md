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
