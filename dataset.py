import os
from glob import glob

import cv2
from PIL import Image
import numpy as np
import torch
import torch.utils.data
from torchvision import transforms as ttf
from torchvision.transforms import functional as F


class CloudOverlapData(torch.utils.data.Dataset):
    def __init__(self, config, root, mode="train", nonempty=False, seed=0):
        self.config = config
        self.root = root
        self.mode = mode
        assert mode in ["train", "eval", "test"]
        self.nonempty = nonempty
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.colorjitter = ColotJitter4Channels(0.2, 0.2, 0.2, 0.1)
        self.resize = ttf.Resize((384, 384))

        if mode == "train":
            if nonempty:
                with open(os.path.join(root, "nonempty_20percent_list.txt"), "r") as f:
                    self.ids = f.read().splitlines()
            else:
                with open(os.path.join(root, "train_list.txt"), "r") as f:
                    self.ids = f.read().splitlines()
        elif mode == "eval":
            with open(os.path.join(root, "eval_list.txt"), "r") as f:
                self.ids = f.read().splitlines()
        elif mode == "test":
            imgs = os.listdir(os.path.join(root, "images"))
            self.ids = sorted(list(map(lambda x: x.replace("patch_", "").replace(".TIF", ""), imgs)))
    
    def __getitem__(self, idx):
        _id = self.ids[idx]
        fp_img = os.path.join(self.root, "images", "patch_" + _id + ".TIF")
        img = cv2.imread(fp_img, -1)
        if self.mode in ["train", "eval"]:
            fp_gt = os.path.join(self.root, "masks", "gt_" + _id + ".TIF")
            gt = cv2.imread(fp_gt, -1) / 255.
            # transform
            img, gt = torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(gt).unsqueeze(0)
            img, gt = self.resize(img), self.resize(gt)
            # random rotation
            rand_rot = torch.randint(0, 4, (1,)).item()
            if rand_rot:
                img = F.rotate(img, rand_rot * 90.)
                gt = F.rotate(gt, rand_rot * 90.)
            # random hflip
            rand_flip = torch.randint(0, 2, (1,)).item()
            if rand_flip:
                img = F.hflip(img)
                gt = F.hflip(gt)
            # random color jitter
            if not self.config["exclude_colorjitter"]:
                rand_color = torch.randint(0, 2, (1,)).item()
                if rand_color:
                    img = self.colorjitter(img)
            return img, gt, fp_img
        elif self.mode == "test":
            img = torch.from_numpy(img).permute(2, 0, 1)
            return img, _id

    def __len__(self):
        return len(self.ids)


class ColotJitter4Channels(torch.nn.Module):
    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float):
        super().__init__()
        self.brightness = [1 - brightness, 1 + brightness]
        self.contrast = [1 - contrast, 1 + contrast]
        self.saturation = [1 - saturation, 1 + saturation]
        self.hue = [0 - hue, 0 + hue]
    
    def forward(self, img):
        b = float(torch.empty(1).uniform_(self.brightness[0], self.brightness[1]))
        c = float(torch.empty(1).uniform_(self.contrast[0], self.contrast[1]))
        s = float(torch.empty(1).uniform_(self.saturation[0], self.saturation[1]))
        h = float(torch.empty(1).uniform_(self.hue[0], self.hue[1]))
        order = torch.randperm(4)

        for oi in order:
            if oi == 0:
                img = self.perform(img, F.adjust_brightness, b)
            elif oi == 1:
                img = self.perform(img, F.adjust_contrast, c)
            elif oi == 2:
                # !! Note: 1-channel saturation adjustment is identity mapping - need some ideas
                img = self.perform(img, F.adjust_saturation, s)
            elif oi == 3:
                # !! Note: 1-channel hue adjustment is identity mapping - need some ideas
                img = self.perform(img, F.adjust_hue, h)

        return img
    
    def perform(self, x, func, factor):
        new_x = []
        for _x in x:
            new_x.append(func(_x.unsqueeze(0), factor))
        return torch.cat(new_x, dim=0)


class CloudData(torch.utils.data.Dataset):
    def __init__(self, config, root, transforms=None, mode="train", nonempty_38=None, nonempty_95=None, include_95=False,
                include_nir=False, bit=8, path_38=None, path_95=None, seed=0):
        self.config = config
        self.root = root
        self.transforms = transforms
        assert mode in ["train", "eval", "test"]
        self.mode = mode
        self.nonempty_38 = nonempty_38
        self.nonempty_95 = nonempty_95
        self.include_95 = include_95
        self.include_nir = include_nir
        assert bit in [8, 16]
        self.getitem = self.getitem_8bit if bit == 8 else self.getitem_16bit
        self.bit = bit
        self.path_38 = path_38 if path_38 is not None else os.path.join(root, "38Cloud")
        self.path_95 = path_95 if path_95 is not None else os.path.join(root, "95Cloud")
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.colorjitter = ColotJitter4Channels(0.2, 0.2, 0.2, 0.1)
        self.resize = ttf.Resize((384, 384))
        self.normalize = ttf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        # list of 38Cloud
        if mode == "train":
            with open(os.path.join(self.path_38, "train", "train_patch_list.txt"), "r") as f:
                self.train_names_38 = f.read().splitlines()
            if bit == 8:
                # use pre-computed png files if 8it and non-nir
                self.images_dir_38 = os.path.join(self.path_38, "train", "rgb_train", "images")
                self.gts_dir_38 = os.path.join(self.path_38, "train", "rgb_train", "masks")
            elif bit == 16:
                # r, g, b paths
                self.red_dir_38 = os.path.join(self.path_38, "train", "train_red")
                self.green_dir_38 = os.path.join(self.path_38, "train", "train_green")
                self.blue_dir_38 = os.path.join(self.path_38, "train", "train_blue")
                self.gts_dir_38 = os.path.join(self.path_38, "train", "train_gt")
            # nir files if required
            if include_nir:
                self.nir_dir_38 = os.path.join(self.path_38, "train", "train_nir")
        elif mode == "eval":
            with open(os.path.join(self.path_38, "train", "eval_patch_list.txt"), "r") as f:
                self.eval_names = f.read().splitlines()
            if bit == 8:
                self.images_dir = os.path.join(self.path_38, "train", "rgb_eval", "images")
                self.gts_dir = os.path.join(self.path_38, "train", "rgb_eval", "masks")
            elif bit == 16:
                self.red_dir = os.path.join(self.path_38, "train", "train_red")
                self.green_dir = os.path.join(self.path_38, "train", "train_green")
                self.blue_dir = os.path.join(self.path_38, "train", "train_blue")
                self.gts_dir = os.path.join(self.path_38, "train", "train_gt")
            if include_nir:
                self.nir_dir = os.path.join(self.path_38, "train", "train_nir")
        elif mode == "test":  # No gt return, computing scores on test set must done in a separate code
            with open(os.path.join(self.path_38, "test", "test_patch_list.txt"), "r") as f:
                self.test_names = f.read().splitlines()
            self.red_dir = os.path.join(self.path_38, "test", "test_red")
            self.green_dir = os.path.join(self.path_38, "test", "test_green")
            self.blue_dir = os.path.join(self.path_38, "test", "test_blue")
            if include_nir:
                self.nir_dir = os.path.join(self.path_38, "test", "test_nir")

        # (optional) add 95Cloud
        if include_95:
            with open(os.path.join(self.path_95, "95-cloud_training_only_additional_to38-cloud", "95_patch_list.txt"), "r") as f:
                self.train_names_95 = f.read().splitlines()
            if bit == 8:
                self.images_dir_95 = os.path.join(self.path_95, "train_all", "images")
                self.gts_dir_95 = os.path.join(self.path_95, "train_all", "masks")
            elif bit == 16:
                self.red_dir_95 = os.path.join(self.path_95, "95-cloud_training_only_additional_to38-cloud", "train_red_additional_to38cloud")
                self.green_dir_95 = os.path.join(self.path_95, "95-cloud_training_only_additional_to38-cloud", "train_green_additional_to38cloud")
                self.blue_dir_95 = os.path.join(self.path_95, "95-cloud_training_only_additional_to38-cloud", "train_blue_additional_to38cloud")
                self.gts_dir_95 = os.path.join(self.path_95, "95-cloud_training_only_additional_to38-cloud", "train_gt_additional_to38cloud")
            if include_nir:
                self.nir_dir_95 = os.path.join(self.path_95, "95-cloud_training_only_additional_to38-cloud", "train_nir_additional_to38cloud")

        # filter out nonempty files
        if nonempty_38 is not None:
            assert mode == "train", "Currently only supports train mode"
            with open(nonempty_38, "r") as f:
                nonempty_files_38 = f.read().splitlines()
            self.train_names_38 = sorted(list(set(self.train_names_38).intersection(set(nonempty_files_38))))
        if (nonempty_95 is not None) and include_95:
            assert mode == "train", "Currently only supports train mode"
            with open(nonempty_95, "r") as f:
                nonempty_files_95 = f.read().splitlines()
            self.train_names_95 = sorted(list(set(self.train_names_95).intersection(set(nonempty_files_95))))

    # use pre-computed images
    def getitem_8bit(self, idx):
        if self.mode == "train":
            rands = []
            # paths
            if idx < self.len_38:  # get data from 38Cloud
                name = self.train_names_38[idx]
                img_fp = os.path.join(self.images_dir_38, "RGB_" + name + ".png")
                gt_fp = os.path.join(self.gts_dir_38, "gt_" + name + ".png")
                if self.include_nir:
                    nir_fp = os.path.join(self.nir_dir_38, "nir_" + name + ".TIF")
            else:  # get data from 95Cloud
                name = self.train_names_95[idx - self.len_38]
                img_fp = os.path.join(self.images_dir_95, "RGB_" + name + ".png")
                gt_fp = os.path.join(self.gts_dir_95, "gt_" + name + ".png")
                if self.include_nir:
                    nir_fp = os.path.join(self.nir_dir_95, "nir_" + name + ".TIF")
            # read
            img = cv2.imread(img_fp, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(gt_fp, cv2.IMREAD_UNCHANGED)
            gt = (gt / 255).astype(np.float32)
            if self.include_nir:
                nir = cv2.imread(nir_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
                img = np.concatenate([img, nir[:, :, None]], axis=-1)
            # transform
            if self.transforms is not None:
                img, gt = self.transforms(img, gt)
            else:
                # to tensor, resize
                img, gt = torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(gt).unsqueeze(0)
                img, gt = self.resize(img), self.resize(gt)
                # random rotation
                rand_rot = torch.randint(0, 4, (1,)).item()
                rands.append(rand_rot)
                if rand_rot:
                    img = F.rotate(img, rand_rot * 90.)
                    gt = F.rotate(gt, rand_rot * 90.)
                # random hflip
                rand_flip = torch.randint(0, 2, (1,)).item()
                rands.append(rand_flip)
                if rand_flip:
                    img = F.hflip(img)
                    gt = F.hflip(gt)
                # random color jitter
                if not self.config["exclude_colorjitter"]:
                    rand_color = torch.randint(0, 2, (1,)).item()
                    rands.append(rand_color)
                    if rand_color:
                        img = self.colorjitter(img)
            return img, gt, (img_fp, rands)
        elif self.mode == "eval":
            # paths
            name = self.eval_names[idx]
            img_fp = os.path.join(self.images_dir, "RGB_" + name + ".png")
            gt_fp = os.path.join(self.gts_dir, "gt_" + name + ".png")
            if self.include_nir:
                nir_fp = os.path.join(self.nir_dir, "nir_" + name + ".TIF")
            # read
            img = cv2.imread(img_fp, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(gt_fp, cv2.IMREAD_UNCHANGED)
            gt = (gt / 255).astype(np.float32)
            if self.include_nir:
                nir = cv2.imread(nir_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
                img = np.concatenate([img, nir[:, :, None]], axis=-1)
            # transform, nothing as default
            if self.transforms is not None:
                img, gt = self.transforms(img, gt)
            else:
                img, gt = torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(gt).unsqueeze(0)
                img, gt = self.resize(img), self.resize(gt)
            # return
            return img, gt, img_fp
        elif self.mode == "test":
            # paths
            name = self.test_names[idx]
            red_fp = os.path.join(self.red_dir, "red_" + name + ".TIF")
            green_fp = os.path.join(self.green_dir, "green_" + name + ".TIF")
            blue_fp = os.path.join(self.blue_dir, "blue_" + name + ".TIF")
            if self.include_nir:
                nir_fp = os.path.join(self.nir_dir, "nir_" + name + ".TIF")
            # read
            img_r = cv2.imread(red_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
            img_g = cv2.imread(green_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
            img_b = cv2.imread(blue_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)

            # ?? act as 8bit png input ?? -> maybe saving and loading required
            img_r, img_g, img_b = img_r * 255, img_g * 255, img_b * 255
            img_r, img_g, img_b = img_r / 255, img_g / 255, img_b / 255

            img = np.stack([img_r, img_g, img_b], axis=0)
            if self.include_nir:
                nir = cv2.imread(nir_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
                img = np.concatenate([img, nir[None, :, :]], axis=0)
            # transform, nothing as default
            if self.transforms is not None:
                img = self.transforms(img)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = self.resize(img)
            # return
            return img, red_fp
        else:
            raise Exception("Not supported mode: {}".format(self.mode))

    # read from original 16-bit TIF images
    def getitem_16bit(self, idx):
        if self.mode == "train":
            # paths
            if idx < self.len_38:  # get data from 38Cloud
                name = self.train_names_38[idx]
                red_fp = os.path.join(self.red_dir_38, "red_" + name + ".TIF")
                green_fp = os.path.join(self.green_dir_38, "green_" + name + ".TIF")
                blue_fp = os.path.join(self.blue_dir_38, "blue_" + name + ".TIF")
                gt_fp = os.path.join(self.gts_dir_38, "gt_" + name + ".TIF")
                if self.include_nir:
                    nir_fp = os.path.join(self.nir_dir_38, "nir_" + name + ".TIF")
            else:  # get data from 95Cloud
                name = self.train_names_95[idx - self.len_38]
                red_fp = os.path.join(self.red_dir_95, "red_" + name + ".TIF")
                green_fp = os.path.join(self.green_dir_95, "green_" + name + ".TIF")
                blue_fp = os.path.join(self.blue_dir_95, "blue_" + name + ".TIF")
                gt_fp = os.path.join(self.gts_dir_95, "gt_" + name + ".TIF")
                if self.include_nir:
                    nir_fp = os.path.join(self.nir_dir_95, "nir_" + name + ".TIF")
            # read
            img_r = cv2.imread(red_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
            img_g = cv2.imread(green_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
            img_b = cv2.imread(blue_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
            img = np.stack([img_r, img_g, img_b], axis=0)
            gt = cv2.imread(gt_fp, cv2.IMREAD_UNCHANGED)
            gt = (gt / 255).astype(np.float32)
            if self.include_nir:
                nir = cv2.imread(nir_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
                img = np.concatenate([img, nir[None, :, :]], axis=0)
            # transform
            if self.transforms is not None:
                img, gt = self.transforms(img, gt)
            else:
                img, gt = torch.from_numpy(img), torch.from_numpy(gt).unsqueeze(0)
                img, gt = self.resize(img), self.resize(gt)
                # random rotation
                rand_rot = torch.randint(0, 4, (1,)).item()
                if rand_rot:
                    img = F.rotate(img, rand_rot * 90.)
                    gt = F.rotate(gt, rand_rot * 90.)
                # random hflip
                rand_flip = torch.randint(0, 2, (1,)).item()
                if rand_flip:
                    img = F.hflip(img)
                    gt = F.hflip(gt)
                # random color jitter
                if not self.config["exclude_colorjitter"]:
                    rand_color = torch.randint(0, 2, (1,)).item()
                    if rand_color:
                        img = self.colorjitter(img)
            return img, gt, red_fp
        elif self.mode == "eval":
            # paths
            name = self.eval_names[idx]
            red_fp = os.path.join(self.red_dir, "red_" + name + ".TIF")
            green_fp = os.path.join(self.green_dir, "green_" + name + ".TIF")
            blue_fp = os.path.join(self.blue_dir, "blue_" + name + ".TIF")
            gt_fp = os.path.join(self.gts_dir, "gt_" + name + ".TIF")
            if self.include_nir:
                nir_fp = os.path.join(self.nir_dir, "nir_" + name + ".TIF")
            # read
            img_r = cv2.imread(red_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
            img_g = cv2.imread(green_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
            img_b = cv2.imread(blue_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
            img = np.stack([img_r, img_g, img_b], axis=-1)
            gt = cv2.imread(gt_fp, cv2.IMREAD_UNCHANGED)
            gt = (gt / 255).astype(np.float32)
            if self.include_nir:
                nir = cv2.imread(nir_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
                img = np.concatenate([img, nir[:, :, None]], axis=-1)
            # transform, nothing as default
            if self.transforms is not None:
                img, gt = self.transforms(img, gt)
            else:
                img, gt = torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(gt).unsqueeze(0)
                img, gt = self.resize(img), self.resize(gt)
            # return
            return img, gt, red_fp
        elif self.mode == "test":
            # paths
            name = self.test_names[idx]
            red_fp = os.path.join(self.red_dir, "red_" + name + ".TIF")
            green_fp = os.path.join(self.green_dir, "green_" + name + ".TIF")
            blue_fp = os.path.join(self.blue_dir, "blue_" + name + ".TIF")
            if self.include_nir:
                nir_fp = os.path.join(self.nir_dir, "nir_" + name + ".TIF")
            # read
            img_r = cv2.imread(red_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
            img_g = cv2.imread(green_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
            img_b = cv2.imread(blue_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
            img = np.stack([img_r, img_g, img_b], axis=-1)
            if self.include_nir:
                nir = cv2.imread(nir_fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / (2 ** 16 - 1)
                img = np.concatenate([img, nir[:, :, None]], axis=-1)
            # transform, nothing as default
            if self.transforms is not None:
                img = self.transforms(img)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = self.resize(img)
            # return
            return img, red_fp
        else:
            raise Exception("Not supported mode: {}".format(self.mode))

    def __getitem__(self, idx):
        return self.getitem(idx)

    def __len__(self):
        if self.mode == "train":
            self.len_38 = len(self.train_names_38)
            if self.include_95:
                _len = self.len_38 + len(self.train_names_95)
            else:
                _len = self.len_38
            return _len
        elif self.mode == "eval":
            return len(self.eval_names)
        elif self.mode == "test":
            return len(self.test_names)
        else:
            raise Exception("Not supported mode")
