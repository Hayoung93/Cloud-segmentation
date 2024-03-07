import torch
from torchvision import transforms
import os
import argparse
from PIL import Image
from tqdm import tqdm

from metrics import iou_score
from archs import NestedUNet
from resnet50_unetpp import UNetWithResnet50Encoder
from attention_unet import AttentionUNet, init_weights
from nafnet import NAFNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help=".pth checkpoint path")
    parser.add_argument("--test_dir", type=str, help="root dir that contains images&masks dirs")
    parser.add_argument("--arch", choices=["v1", "v2", "v3", "v4", "v4-1"], type=str, help="network architecture, v1=vgg-unet, v2=resnet50-unet")
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()
    return args


def main(args):
    if args.arch == "v1":
        model = NestedUNet(1)
    elif args.arch == "v2":
        model = UNetWithResnet50Encoder(1)
    elif args.arch == "v3":
        model = AttentionUNet(args.input_channels, 1)
        model = init_weights(model, "kaiming")
    elif "v4" in args.arch:
        if args.arch == "v4":
            model = NAFNet(img_channel=args.input_channels, out_channel=1, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
        elif args.arch == "v4-1":
            model = NAFNet(img_channel=args.input_channels, out_channel=1, width=32, middle_blk_num=1, enc_blk_nums=[3, 4, 6, 3], dec_blk_nums=[3, 4, 6, 3])
    else:
        raise Exception("Not supported network architecture")
    cp = torch.load(args.checkpoint)
    # model.load_state_dict(cp)
    model.load_state_dict(cp["state_dict"])
    model = model.eval()
    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    totensor = transforms.ToTensor()
    resize = transforms.Resize((384, 384))
    root = args.test_dir
    files = os.listdir(os.path.join(root, "images"))
    # # filter empty images - only for 38cloud and 95cloud
    # nonempty_fp = "/data/data/inspyrenet-datasets/38Cloud/38-Cloud_training/rgb_train/nonempty.txt"
    # with open(nonempty_fp, "r") as f:
    #     nonempty = f.read().splitlines()
    # nonempty_set = set(nonempty)
    # val_set = set(map(lambda x: x[4:-4], files))
    # val_set = list(val_set.intersection(nonempty_set))
    # files = list(map(lambda x: "RGB_" + x + ".png", val_set))

    ious = []
    pbar = tqdm(files)
    for img_fn in pbar:
        img = Image.open(os.path.join(root, "images", img_fn)).convert("RGB")
        # img = resize(img)
        img = normalize(totensor(img)).unsqueeze(0).cuda().to(torch.float32)
        # gt = Image.open(os.path.join(root, "masks", img_fn.replace("RGB", "gt")))
        gt = Image.open(os.path.join(root, "masks", "edited_corrected_gts_" + img_fn))
        gt = resize(gt)
        gt = totensor(gt).unsqueeze(0).cuda().to(torch.float32)

        out = model(img)
        if args.out_dir != "":
            pass
        score = iou_score(out, gt)
        ious.append(score)
        pbar.set_description("score: {:.4f}".format(score))
        pbar.update(1)
    pbar.close()
    ious = torch.tensor(ious)
    print(ious)
    print(ious.mean())


if __name__ == "__main__":
    args = get_args()
    main(args)
