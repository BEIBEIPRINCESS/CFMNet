import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu
from PIL import Image
import random

from .transform import Compose, RandomScale, SmartCropV1


CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest',
           'agricultural')

PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
           [159, 129, 183], [0, 255, 0], [255, 195, 128]]


ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (1024, 1024)
TEST_IMG_SIZE = (1024, 1024)


def get_training_transform():
    train_transform = [
        # albu.Resize(height=1024, width=1024),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        # albu.RandomRotate90(p=0.5),
        # albu.OneOf([
        #     albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
        #     albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25)
        # ], p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    # multi-scale training and crop
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=255, nopad=False)])
    img, mask = crop_aug(img, mask)

    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        # albu.Resize(height=1024, width=1024, interpolation=cv2.INTER_CUBIC),
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


class LoveDATrainDataset(Dataset):
    def __init__(self, data_root, img_dir='images_png', mosaic_ratio=0.25,
                 mask_dir='masks_png_convert', img_suffix='.png', mask_suffix='.png',
                 transform=train_aug, img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mosaic_ratio = mosaic_ratio

        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        img, mask = self.load_img_and_mask(index)
        if p_ratio < self.mosaic_ratio:
            img, mask = self.load_mosaic_img_and_mask(index)
        if self.transform:
            img, mask = self.transform(img, mask)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id, img_type = self.img_ids[index]
        results = {'img': img, 'gt_semantic_seg': mask, 'img_id': img_id, 'img_type': img_type}

        return results

    def __len__(self):
        length = len(self.img_ids)
        return length

    def get_img_ids(self, data_root, img_dir, mask_dir):
        urban_img_filename_list = os.listdir(osp.join(data_root, 'Urban', img_dir))
        urban_mask_filename_list = os.listdir(osp.join(data_root, 'Urban', mask_dir))
        assert len(urban_img_filename_list) == len(urban_mask_filename_list)
        urban_img_ids = [(str(id.split('.')[0]), 'Urban') for id in urban_img_filename_list]

        rural_img_filename_list = os.listdir(osp.join(data_root, 'Rural', img_dir))
        rural_mask_filename_list = os.listdir(osp.join(data_root, 'Rural', mask_dir))
        assert len(rural_img_filename_list) == len(rural_mask_filename_list)
        rural_img_ids = [(str(id.split('.')[0]), 'Rural') for id in rural_img_filename_list]
        img_ids = urban_img_ids + rural_img_ids

        return img_ids

    def load_img_and_mask(self, index):
        img_id, img_type = self.img_ids[index]
        img_name = osp.join(self.data_root, img_type, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, img_type, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')

        return img, mask

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        return img, mask


class LoveDATestDataset(Dataset):
    def __init__(self, data_root, img_dir='images_png',
                 img_suffix='.png',  mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir

        self.img_suffix = img_suffix
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir)

    def __getitem__(self, index):
        img = self.load_img(index)

        img = np.array(img)
        aug = albu.Normalize()(image=img)
        img = aug['image']

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img_id, img_type = self.img_ids[index]

        results = {'img': img, 'img_id': img_id, 'img_type': img_type}

        return results

    def __len__(self):
        length = len(self.img_ids)

        return length

    def get_img_ids(self, data_root, img_dir):
        urban_img_filename_list = os.listdir(osp.join(data_root, 'Urban', img_dir))
        urban_img_ids = [(str(id.split('.')[0]), 'Urban') for id in urban_img_filename_list]
        rural_img_filename_list = os.listdir(osp.join(data_root, 'Rural', img_dir))
        rural_img_ids = [(str(id.split('.')[0]), 'Rural') for id in rural_img_filename_list]
        img_ids = urban_img_ids + rural_img_ids

        return img_ids

    def load_img(self, index):
        img_id, img_type = self.img_ids[index]
        img_name = osp.join(self.data_root, img_type, self.img_dir, img_id + self.img_suffix)
        img = Image.open(img_name).convert('RGB')

        return img
