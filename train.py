import argparse
import os
import random
from collections import OrderedDict
from glob import glob
import shutil

import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import archs
import losses
from dataset import Dataset, CloudData, CloudOverlapData
from metrics import iou_score
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

from resnet50_unetpp import UNetWithResnet50Encoder
from attention_unet import AttentionUNet, init_weights
from nafnet import NAFNet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument("--save_per", type=int, default=10)

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='38cloud_16bit')
    parser.add_argument("--train_dir", type=str)
    parser.add_argument("--eval_dir", type=str)
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', "AdamW", 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument("--out_dir", type=str, default="", help="for saving val/test img")
    parser.add_argument('--resume', type=str)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--pretrain", type=str, default="imagenet1k")
    parser.add_argument("--include_nir", action="store_true")
    parser.add_argument("--exclude_colorjitter", action="store_true")
    parser.add_argument("--dataparallel", action="store_true")
    parser.add_argument("--record_file", type=str, default="")

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        # if config["record_file"] != "":
        #     param_cumsum = 0
        #     for n, p in model.named_parameters():
        #         param_cumsum += p.sum()
        #     with open(config["record_file"], "a") as f:
        #         f.write(str(param_cumsum.item()) + "\n")
        #     with open(config["record_file"].replace(".txt", "_rands.txt"), "a") as f:
        #         f.write(str(_[1]) + "\n")

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.inference_mode():
        pbar = tqdm(total=len(val_loader))
        for input, target, img_fp in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                raise Exception("Deep supervision is currently not supported")
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                avg_meters['loss'].update(loss.item(), input.size(0))
                for i, (out, tar) in enumerate(zip(output, target)):
                    # if config["out_dir"] != "":
                    #     shutil.copyfile(img_fp[i], os.path.join(config["out_dir"], "inputs", img_fp[i].split("/")[-1]))
                    #     save_image((out.sigmoid() > 0.5).to(torch.float), os.path.join(config["out_dir"], "outputs", img_fp[i].split("/")[-1].replace(".TIF", ".png")))
                    iou = iou_score(out, tar)
                    avg_meters['iou'].update(iou)

            # avg_meters['loss'].update(loss.item(), input.size(0))
            # avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def test(config, loader, model):
    # switch to evaluate mode
    model.eval()
    if config["out_dir"] != "":
        os.makedirs(config["out_dir"], exist_ok=True)

    with torch.inference_mode():
        pbar = tqdm(total=len(loader))
        for input, img_fp in loader:
            input = input.cuda()

            # compute output
            output = model(input)
            if config["out_dir"] != "":
                if config["dataset"] in ["38cloud_16bit", "38cloud_8bit", "95cloud_16bit", "95cloud_8bit"]:
                    fp_split = img_fp[0].split("_")
                    by_ind = fp_split.index("by")
                    _id = "_".join(fp_split[by_ind - 2:]).replace(".TIF", "")
                elif config["dataset"] == "cloud_overlap":
                    _id = img_fp[0]
                # save_image(output, os.path.join(config["out_dir"], _id + ".png"))
                torch.save(output[0][0], os.path.join(config["out_dir"], _id + ".pt"))
            # save
            pbar.update(1)
        pbar.close()



def main():
    config = vars(parse_args())

    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    if config["arch"] in ["v1", "NestedUNet"]:
        model = archs.NestedUNet(config['num_classes'], config['input_channels'], config['deep_supervision'])
    elif config["arch"] in ["v2"]:
        model = UNetWithResnet50Encoder(n_classes=config["num_classes"], pretrain=config["pretrain"])
    elif config["arch"] in ["v3"]:
        model = AttentionUNet(config["input_channels"], config["num_classes"])
        model = init_weights(model, "kaiming")
    elif "v4" in config["arch"]:
        if config["arch"] == "v4":  # original nafnet layers
            model = NAFNet(img_channel=config["input_channels"], out_channel=config["num_classes"], width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
        elif config["arch"] == "v4-1":  # mimic resnet34 layers
            model = NAFNet(img_channel=config["input_channels"], out_channel=config["num_classes"], width=32, middle_blk_num=1, enc_blk_nums=[3, 4, 6, 3], dec_blk_nums=[3, 4, 6, 3])
        elif config["arch"] == "v4-1-1":  # mimic resnet34 layers
            model = NAFNet(img_channel=config["input_channels"], out_channel=config["num_classes"], width=32, middle_blk_num=1, enc_blk_nums=[3, 4, 6, 3], dec_blk_nums=[3, 6, 4, 3])
        elif config["arch"] == "v4-1-2":
            model = NAFNet(img_channel=config["input_channels"], out_channel=config["num_classes"], width=32, middle_blk_num=1, enc_blk_nums=[2, 4, 6, 8], dec_blk_nums=[8, 6, 4, 2])
        elif config["arch"] == "v4-2":  # mimic resnet34 + skip connection with input before last conv
            model = NAFNet(img_channel=config["input_channels"], out_channel=config["num_classes"], width=32, middle_blk_num=1, enc_blk_nums=[3, 4, 6, 3], dec_blk_nums=[3, 4, 6, 3], last_connection=True)
    else:
        raise Exception("Not supported architecture")

    model = model.cuda()
    if config["dataparallel"]:
        model = torch.nn.DataParallel(model)

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(params, config["lr"], weight_decay=config["weight_decay"])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError


    best_iou = 0
    best_epoch = -1
    trigger = 0
    start_epoch = 0
    if config['resume'] is not None and config['resume'] != '' and os.path.isfile(config['resume']):
        # load model
        cp = torch.load(config['resume'])
        msg = model.load_state_dict(cp["state_dict"])
        print("Loaded weight: ", msg)
        # load optimizer and scheduler
        optimizer.load_state_dict(cp["optimizer"])
        scheduler.load_state_dict(cp["scheduler"])
        # load others
        best_iou = cp["best_iou"]
        best_epoch = cp["best_epoch"]
        start_epoch = cp["epoch"] + 1

    def seed_worker(worker_id):
        # worker_seed = torch.initial_seed() % 2**32
        worker_seed = worker_id % 2**32
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    if config["dataset"] in ["38cloud_16bit", "38cloud_8bit", "95cloud_16bit", "95cloud_8bit"]:
        # custom transforms that change image and mask in the same way
        # use default transforms if None
        transforms_train = None
        transforms_val = None
        transforms_test = None
        trainset = CloudData(config, "/data/data", transforms_train, "train", "/data/data/38Cloud/train/nonempty.txt",
                            "/data/data/95Cloud/95-cloud_training_only_additional_to38-cloud/nonempty_95.txt",
                            True if config["dataset"][:2] == "95" else False,
                            config["include_nir"], int(config["dataset"].split("_")[1].replace("bit", "")), seed=seed) 
        valset = CloudData(config, "/data/data", transforms_val, "eval", None, None, False,
                            config["include_nir"], int(config["dataset"].split("_")[1].replace("bit", "")), seed=seed)
        if config["test"]:
            testset = CloudData(config, "/data/data", transforms_test, "test", None, None, False,
                            config["include_nir"], int(config["dataset"].split("_")[1].replace("bit", "")), seed=seed)
    elif config["dataset"] == "cloud_overlap":
        trainset = CloudOverlapData(config, "/data/data/38Cloud/train/rgbn_16bit_overlap", "train", True)  # nonempty patches
        valset = CloudOverlapData(config, "/data/data/38Cloud/train/rgbn_16bit_overlap", "eval")
        if config["test"]:
            testset = CloudOverlapData(config, "/data/data/38Cloud/test/rgbn_16bit_overlap", "test")
    else:
        raise Exception("Not supported dataset")
    trainloader = DataLoader(trainset, config["batch_size"], True, num_workers=config["num_workers"],
                            drop_last=False, worker_init_fn=seed_worker, generator=g)
    valloader = DataLoader(valset, 1, False, num_workers=config["num_workers"], drop_last=False)
    if config["test"]:
        testloader = DataLoader(testset, 1, False, num_workers=config["num_workers"], drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])

    if config['eval']:
        val_log = validate(config, valloader, model, criterion)
        print(val_log)
        exit(0)
    
    if config["test"]:
        test(config, testloader, model)
        exit(0)

    for epoch in range(start_epoch, config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, trainloader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, valloader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            best_iou = val_log['iou']
            best_epoch = epoch
            trigger = 0
            save_dict = {
                "state_dict": model.module.state_dict() if config["dataparallel"] else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_iou": best_iou,
                "best_epoch": best_epoch,
                "epoch": epoch
            }
            torch.save(save_dict, 'models/%s/model_best.pth' % config['name'])
            print("=> saved best model")
        if (epoch + 1) % config["save_per"] == 0:
            save_dict = {
                "state_dict": model.module.state_dict() if config["dataparallel"] else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_iou": best_iou,
                "best_epoch": best_epoch,
                "epoch": epoch
            }
            torch.save(save_dict, 'models/{}/model_{:03d}.pth'.format(config['name'], epoch))

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

        # save last status for resume
        save_dict = {
            "state_dict": model.module.state_dict() if config["dataparallel"] else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_iou": best_iou,
            "best_epoch": best_epoch,
            "epoch": epoch
        }
        torch.save(save_dict, 'models/{}/model.pth'.format(config['name']))
    print("Best validation IoU: {} ({})".format(best_iou, best_epoch))

if __name__ == '__main__':
    main()
