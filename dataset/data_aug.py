import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


# 图像预处理和数据增强
class ResizeTransform:
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Resize image and label
        image = transforms.Resize(self.output_size)(image)
        label = transforms.Resize(self.output_size)(label)

        return {'image': image, 'label': label}


class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if torch.rand(1).item() < self.flip_prob:
            image = F.hflip(image)
            label = F.hflip(label)

        return {'image': image, 'label': label}


class RandomVerticalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if torch.rand(1).item() < self.flip_prob:
            image = F.vflip(image)
            label = F.vflip(label)

        return {'image': image, 'label': label}


class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        angle = transforms.RandomRotation.get_params([-self.degrees, self.degrees])

        image = F.rotate(image, angle)
        label = F.rotate(label, angle)

        return {'image': image, 'label': label}


class RandomCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.output_size)

        image = F.crop(image, i, j, h, w)
        label = F.crop(label, i, j, h, w)

        return {'image': image, 'label': label}


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        image = self.transform(image)

        return {'image': image, 'label': label}


class ToTensorTransform:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Convert PIL Images to Tensors
        image = transforms.ToTensor()(image)
        label = transforms.ToTensor()(label)

        return {'image': image, 'label': label}


# 合并预处理和数据增强
#data_transforms = transforms.Compose([
#    ResizeTransform((256, 256)),
#    RandomHorizontalFlip(),
#    RandomVerticalFlip(),
#    RandomRotation(10),
#    RandomCrop((224, 224)), # 随机裁剪到224x224尺寸
#    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#    ToTensorTransform()
#])

def get_transform(args):
    resize_size = args['resize_size']
    crop_size = args['crop_size']
    flip_h = args['flip_h']
    flip_v = args['flip_v']
    rotate_angle = args['rotate_angle']
    jitter = args['jitter']

    transform_list = []
    transform_list.append(ResizeTransform((resize_size, resize_size)))
    if flip_h:
        transform_list.append(RandomHorizontalFlip())
    if flip_v:
        transform_list.append(RandomVerticalFlip())
    if rotate_angle > 0:
        transform_list.append(RandomRotation(rotate_angle))
    transform_list.append(RandomCrop((crop_size, crop_size)))
    if jitter:
        transform_list.append(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    transform_list.append(ToTensorTransform())

    return transforms.Compose(transform_list)
        
