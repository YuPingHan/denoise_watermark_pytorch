import os
import sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import math

from common_utils import norm_homedir

cur_path = os.path.abspath(__file__)
proj_root_path = os.path.dirname(os.path.dirname(cur_path))
sys.path.append(proj_root_path)
from utils import load_imgs, img2patch, pil2patch


class DenoisePatchDataset(Dataset):
    def __init__(self, data_list, load_mode, transform=None, patch_size=50, overlap=30):
        data_list = open(data_list).read().splitlines()
        ori_imgs = []
        cleaned_imgs = []
        ic = 1 if load_mode == 'gray' else 3 # input_channel
        for row in data_list:
            ori_img_path, cleaned_img_path = row.split('\t')
            ori_img_path = norm_homedir(ori_img_path)
            cleaned_img_path = norm_homedir(cleaned_img_path)
            if load_mode == 'gray':
                #ori_img = Image.open(ori_img_path).convert('L')
                ori_img = cv2.imread(ori_img_path, cv2.IMREAD_GRAYSCALE)
            else:
                #ori_img = Image.open(ori_img_path).convert('RGB')
                ori_img = cv2.imread(ori_img_path, cv2.IMREAD_COLOR)
                ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            if ori_img is None:
                continue
            ori_imgs.append(ori_img)

            if load_mode == 'gray':
                #cleaned_img = Image.open(cleaned_img_path).convert('L')
                cleaned_img = cv2.imread(cleaned_img_path, cv2.IMREAD_GRAYSCALE)
            else:
                #cleaned_img = Image.open(cleaned_img_path).convert('RGB')
                cleaned_img = cv2.imread(cleaned_img_path, cv2.IMREAD_COLOR)
                cleaned_img = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB)
            cleaned_imgs.append(cleaned_img)

        ori_data = img2patch(ori_imgs, patch_size, overlap)
        cleaned_data = img2patch(cleaned_imgs, patch_size, overlap)
        ori_data = np.array(ori_data)
        cleaned_data = np.array(cleaned_data)

        assert ori_data.shape == cleaned_data.shape, print(f'ori_data.shape: {ori_data.shape}, cleaned_data.shape: {cleaned_data.shape}')

        self.ori_data = ori_data
        self.cleaned_data = cleaned_data
        self.transform = transform

    def __getitem__(self, idx):
        ori_patch = self.ori_data[idx]
        cleaned_patch = self.cleaned_data[idx]
        sample = {'image': Image.fromarray(ori_patch), 'label': Image.fromarray(cleaned_patch)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.ori_data)


class DenoiseImageDataset(Dataset):
    def __init__(self, data_list, load_mode, transform=None):
        data_list = open(data_list).read().splitlines()
        ori_imgs = []
        cleaned_imgs = []
        for row in data_list:
            ori_img_path, cleaned_img_path = row.split('\t')
            ori_img_path = norm_homedir(ori_img_path)
            cleaned_img_path = norm_homedir(cleaned_img_path)
            if load_mode == 'gray':
                ori_img = cv2.imread(ori_img_path, cv2.IMREAD_GRAYSCALE)
            else:
                ori_img = cv2.imread(ori_img_path, cv2.IMREAD_COLOR)
                ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            if ori_img is None:
                continue
            ori_imgs.append(ori_img)

            if load_mode == 'gray':
                cleaned_img = cv2.imread(cleaned_img_path, cv2.IMREAD_GRAYSCALE)
            else:
                cleaned_img = cv2.imread(cleaned_img_path, cv2.IMREAD_COLOR)
                cleaned_img = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB)
            cleaned_imgs.append(cleaned_img)

        assert len(ori_imgs) == len(cleaned_imgs), print(f'ori_imgs len: {len(ori_imgs)}, cleaned_imgs len: {len(cleaned_imgs)}')

        self.ori_imgs = ori_imgs
        self.cleaned_imgs = cleaned_imgs
        self.transform = transform

    def __getitem__(self, idx):
        ori_img = self.ori_imgs[idx]
        cleaned_img = self.cleaned_imgs[idx]
        sample = {'image': Image.fromarray(ori_img), 'label': Image.fromarray(cleaned_img)}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.ori_imgs)


if __name__ == '__main__':
    from dataset.data_aug import *

    data_transforms = transforms.Compose([
        ResizeTransform((256, 256)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(10),
        RandomCrop((224, 224)), # 随机裁剪到224x224尺寸
        #ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensorTransform()
    ]) 

    data_list = '/data4/hanyp/dataset/gen_i2i_watermark_remove/finals/gen_i2i_watermark_remove_val_v0.0.txt'
    #dataset = DenoisePatchDataset(data_list, transform=data_transforms, load_mode='rgb')
    #print(len(dataset))
    #one_sample = dataset[0]
    #print(len(one_sample))
    #print(one_sample['image'].dtype)
    #print(one_sample['image'].shape)

    dataset = DenoiseImageDataset(data_list, transform=data_transforms, load_mode='rgb')
    print(len(dataset))
    one_sample = dataset[0]
    print(len(one_sample))
    print(one_sample['image'].dtype)
    print(one_sample['image'].shape)
