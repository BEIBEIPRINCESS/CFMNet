import os
import torch
import sys
import json
import random
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import math

from models.CFMNet import CFMNet


class RSDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert("RGB")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


# -----------------auto path------------------------
def _normalize_dataset_root(path):
    path = os.path.abspath(os.path.expanduser(path))
    return path if path.endswith(os.sep) else path + os.sep


def _is_dataset_root(path):
    return (
        os.path.isdir(os.path.join(path, "train"))
        and os.path.isdir(os.path.join(path, "val"))
    )


def _select_existing_path(*candidates):
    candidates = [path for path in candidates if path]
    for path in candidates:
        path = _normalize_dataset_root(path)
        if _is_dataset_root(path):
            return path

    searched = "\n".join(f"  - {_normalize_dataset_root(path)}" for path in candidates)
    raise FileNotFoundError(
        "dataset path does not exist or is missing train/ and val/. Searched:\n"
        + searched
    )


def _dataset_candidates(args, *built_in_paths):
    data_path = getattr(args, "data_path", "")
    if data_path:
        return (data_path,)
    return built_in_paths


def resolve_dataset(args):
    dataset = args.dataset
    if args.dataset == 'RESISC45-82':
        data_path = _select_existing_path(*_dataset_candidates(
            args,
            '../data/NWPU-RESISC45-82/',
        ))
        num_classes = 45
    elif args.dataset == 'UCM-82':
        data_path = _select_existing_path(*_dataset_candidates(
            args,
            '../data/UCMerced/',
        ))
        num_classes = 21
    elif args.dataset == 'AID-82':
        data_path = _select_existing_path(*_dataset_candidates(
            args,
            '../data/AID/',
        ))
        num_classes = 30
    else:
        raise ValueError(f"unsupported dataset: {args.dataset}")

    return dataset, data_path, num_classes
# -------------------------------------------------------

def load_model(num_classes, device):
    model = CFMNet(num_classes=num_classes).to(device)
    version = 'CFMNet'
    return model, version


def read_train_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    imagenet_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    imagenet_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(imagenet_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_train_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    every_class_num = []
    supported = [".jpeg", ".jpg", ".JPG", ".png", ".PNG", ".JPEG", ".tif"]

    for cla in imagenet_class:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla]

        every_class_num.append(len(images))

        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    print("{} images for training.".format(len(train_images_path)))
    assert len(train_images_path) > 0, "not find data for train."

    return train_images_path, train_images_label


def read_val_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    imagenet_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    imagenet_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(imagenet_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_val_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpeg", ".jpg", ".JPG", ".png", ".PNG", ".JPEG", ".tif"]

    for cla in imagenet_class:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla]

        every_class_num.append(len(images))

        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images for validation.".format(len(val_images_path)))
    assert len(val_images_path) > 0, "not find data for train."

    return val_images_path, val_images_label


def train_one_epoch(model, optimizer, scheduler, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

import torch.nn as nn

def replace_batchnorm(model: nn.Module, replace_with: str = "identity"):
    """
     BatchNorm 
    replace_with:
      - "identity": BN -> Identity
      - "freeze":   BN eval 
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            if replace_with == "identity":
                setattr(model, name, nn.Identity())
            elif replace_with == "freeze":
                module.eval()
                for p in module.parameters():
                    p.requires_grad_(False)
            else:
                raise ValueError(f"Unknown replace_with: {replace_with}")
        else:
            replace_batchnorm(module, replace_with=replace_with)
