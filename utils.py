import os
import cv2
from PIL import Image
import numpy as np
import math
from math import cos, pi
import ntplib
from datetime import datetime
from time import sleep


def norm_gray(img):
    normed_gray_img = img.copy()
    mask = (img >= 170) & (img <= 250)

    normed_gray_img[mask] = 245

    return normed_gray_img


def load_imgs(img_dir):
    ori_imgs = []
    normed_gray_imgs = []
    names = []

    img_names = os.listdir(img_dir)
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        ori_img = cv2.imread(img_path, 0)
        if ori_img is None:
            continue
        #normed_gray_img = norm_gray(ori_img)
        #ori_imgs.append(ori_img)
        #normed_gray_imgs.append(normed_gray_img)
        ori_imgs.append(ori_img)
        names.append(img_name)

    return ori_imgs, names


def img2patch(imgs, patch_size, overlap):
    data = []
    for i in range(len(imgs)):
        img = imgs[i]
        h, w = img.shape[:2]

        num_x = math.ceil((w - patch_size) / (patch_size - overlap)) + 1
        num_y = math.ceil((h - patch_size) / (patch_size - overlap)) + 1

        for idx_x in range(num_x):
            for idx_y in range(num_y):
                x_start = (patch_size - overlap) * idx_x
                y_start = (patch_size - overlap) * idx_y
                if x_start + patch_size >= w:
                    x_start = w - patch_size
                if y_start + patch_size >= h:
                    y_start = h - patch_size

                data.append(img[y_start : y_start+patch_size, x_start : x_start+patch_size])

    return data


def pil2patch(img_pils, patch_size, overlap, ic):
    imgs = [np.array(img_pil) for img_pil in img_pils]

    data = img2patch(imgs, patch_size, overlap)

    data = np.array(data).reshape((len(data), ic, patch_size, patch_size))

    return data


def hanning(size):
    v = np.divide(np.arange(size), size)
    v2 = np.ones(size)

    v2[0: size] = v
    hanv = - np.cos(v2 * 2 * math.pi) * 0.5 + 0.5

    ret = np.outer(hanv, hanv) + 0.01
    ret = np.divide(ret, ret.sum())

    return ret


def patch2img(data, original_images, patch_size, overlap, c):
    count = 0
    images = []
    window = hanning(patch_size)
    if c != 1:
        window = np.expand_dims(window, 2)
        window = np.tile(window, (1, 1, 3))

    for i in range(0, len(original_images)):
        size_y = original_images[i].shape[0]
        size_x = original_images[i].shape[1]

        if c == 1:
            images.append(np.zeros((size_y, size_x), np.float64))
            weight = np.zeros((size_y, size_x), np.float64)
        else:
            images.append(np.zeros((size_y, size_x, c), np.float64))
            weight = np.zeros((size_y, size_x, c), np.float64)
        num_y = math.ceil((size_y - patch_size) / (patch_size - overlap)) + 1
        num_x = math.ceil((size_x - patch_size) / (patch_size - overlap)) + 1

        for sx in range(0, num_x):
            for sy in range(0, num_y):
                x = (patch_size - overlap) * sx
                y = (patch_size - overlap) * sy
                if x + patch_size >= size_x:
                    x = size_x - patch_size
                if y + patch_size >= size_y:
                    y = size_y - patch_size
                img_copy = images[i]
                #print(f'data[count].shape: {data[count].shape}, window.shape: {window.shape}, img_copy.shape: {img_copy.shape}, weight.shape: {weight.shape}')

                if c == 1:
                    mul = np.multiply(data[count].reshape(patch_size, patch_size), window)
                    img_copy[y:y+patch_size, x:x+patch_size] += mul
                    weight[y: y+patch_size, x:x+patch_size] += window
                    images[i] = img_copy
                else:
                    mul = np.multiply(data[count].reshape(patch_size, patch_size, c), window)
                    img_copy[y:y+patch_size, x:x+patch_size] += mul
                    weight[y: y+patch_size, x:x+patch_size] += window
                    images[i] = img_copy

                count = count + 1

        images[i] = np.divide(images[i], weight)

    return images


def get_ntp_time(server='pool.ntp.org', max_retries=10, delay=1):
    client = ntplib.NTPClient()

    retries = 0
    while retries < max_retries:
        try:
            response = client.request(server)
            print('Success to get ntp time!')

            return datetime.utcfromtimestamp(response.tx_time + 8 * 60 * 60) # 注意要加上8个小时
        except ntplib.NTPException as e:
            print(f'Attempt {retries + 1} failed: {e}')
            retries += 1
            sleep(delay)

    raise Exception(f"Failed to connect to NTP server {server} after {max_retries} attempts")


def get_ds_info(data_list):
    base_name = os.path.splitext(os.path.basename(data_list))[0]
    items = base_name.split('_')
    task_name = '_'.join(items[:4])
    version = base_name.split('_')[-1][1:]

    return task_name, version


def get_train_name(configs, prefix=''):
    date_time = get_ntp_time()
    #date_time = datetime.now()
    time_str = f'{date_time.year}{date_time.month:0>2d}{date_time.day:0>2d}_{date_time.hour:0>2d}{date_time.minute:0>2d}'

    task_name, train_set_version = get_ds_info(configs['train_data_list'])
    _, val_set_version = get_ds_info(configs['val_data_list'])
    #backbone = configs['model_type']
    data_mode = configs['data_mode']
    load_mode = configs['load_mode']
    epochs = configs['epochs']
    bs = configs['batch_size']
    #warmup_epochs = configs['warmup_epochs']
    model_type = configs['model_type']

    train_name = f'{time_str}_{task_name}_train{train_set_version}_val{val_set_version}_{data_mode}_{load_mode}_{model_type}_{epochs}e_{bs}bs'

    return train_name


def log_configs(logfile, configs):
    # log dataset config
    logfile.write(f'train_data_list: {configs["train_data_list"]}\n')
    logfile.write(f'val_data_list: {configs["val_data_list"]}\n')
    logfile.write(f'dataloader_num_workers: {configs["dataloader_num_workers"]}\n')
    logfile.write(f'data_mode: {configs["data_mode"]}\n')
    logfile.write(f'load_mode: {configs["load_mode"]}\n')
    logfile.write(f'patch_size: {configs["patch_size"]}\n')
    logfile.write(f'overlap: {configs["overlap"]}\n\n')

    # data augment config
    logfile.write(f'resize_size: {configs["resize_size"]}\n')
    logfile.write(f'crop_size: {configs["crop_size"]}\n')
    logfile.write(f'flip_h: {configs["flip_h"]}\n')
    logfile.write(f'flip_v: {configs["flip_v"]}\n')
    logfile.write(f'rotate_angle: {configs["rotate_angle"]}\n')
    logfile.write(f'jitter: {configs["jitter"]}\n\n')

    # log model config
    logfile.write(f'model_type: {configs["model_type"]}\n')
    #logfile.write(f'pretrained: {configs["pretrained"]}\n')

    # log train schedule
    logfile.write(f'epochs: {configs["epochs"]}\n')
    logfile.write(f'batch_size: {configs["batch_size"]}\n')
    logfile.write(f'val_batch_size: {configs["val_batch_size"]}\n')
    #logfile.write(f'warmup_epochs: {configs["warmup_epochs"]}\n')
    logfile.write(f'beta_1: {configs["beta_1"]}\n')
    logfile.write(f'beta_2: {configs["beta_2"]}\n')
    logfile.write(f'weight_decay: {configs["weight_decay"]}\n')
    logfile.write(f'clipnorm: {configs["clipnorm"]}\n')

    # log and save
    logfile.write(f'print interval: {configs["print_interval"]}\n')
    train_name = get_train_name(configs)
    logfile_path = os.path.join(configs['logs_dir'], f'{train_name}.log')
    logfile.write(f'log path: {logfile_path}\n')
    save_dir = os.path.join(configs['save_root_dir'], train_name)
    logfile.write(f'save dir: {save_dir}\n\n')

    logfile.flush()


if __name__ == '__main__':
    data_list = 'gen_i2i_watermark_remove_train_v0.0.txt'
    print(get_ds_info(data_list))
