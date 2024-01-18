import torch
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
from metrics import iou_score
from archs import NestedUNet
from resnet50_unetpp import UNetWithResnet50Encoder
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help=".pth checkpoint path")
    parser.add_argument("--test_dir", type=str, help="root dir that contains images&masks dirs")
    parser.add_argument("--arch", choices=["v1", "v2"], type=str, help="network architecture, v1=vgg-unet, v2=resnet50-unet")
    args = parser.parse_args()
    return args


def main(args):
    if args.arch == "v1":
        model = NestedUNet(1)
    elif args.arch == "v2":
        model = UNetWithResnet50Encoder(1)
    else:
        raise Exception("Not supported network architecture")
    cp = torch.load(args.checkpoint)
    model.load_state_dict(cp)
    model = model.eval()
    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    totensor = transforms.ToTensor()
    resize = transforms.Resize((384, 384))
    root = args.test_dir
    files = os.listdir(os.path.join(root, "images"))
    ious = []
    
    pbar = tqdm(files)
    for img_fn in pbar:
        img = Image.open(os.path.join(root, "images", img_fn)).convert("RGB")
        img = resize(img)
        img = normalize(totensor(img)).unsqueeze(0).cuda().to(torch.float32)
        gt = Image.open(os.path.join(root, "masks", img_fn.replace("RGB", "gt")))
        gt = resize(gt)
        gt = totensor(gt).unsqueeze(0).cuda().to(torch.float32)

        out = model(img)
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
