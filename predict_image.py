import os
import sys
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from common_utils import check_dir
from cv_common_utils import IMAGE_FORMATS

cur_path = os.path.abspath(__file__)
proj_root_path = os.path.dirname(cur_path)
sys.path.append(proj_root_path)
from models.cnn import CNN, ResNet
from models.unet import UNet
from utils import load_imgs, img2patch, patch2img


device = 'cuda:3'
bs = 32
cat = True
load_mode = 'gray'
model_type = 'unet'
resize_size = 512 

transform = transforms.Compose([
    transforms.Resize((resize_size, resize_size)),
    transforms.ToTensor()
])

# load imgs
#test_img_dir = '/data4/hanyp/watermark/denoise_watermark/dataset/zyb_imgs_300_val'
#test_img_dir = '/data4/hanyp/watermark/denoise_watermark/dataset/zyb'
test_img_dir = 'test_case'
img_names = os.listdir(test_img_dir)

imgs = []
for img_name in img_names:
    if os.path.splitext(img_name)[1] not in IMAGE_FORMATS:
        continue
    img_path = os.path.join(test_img_dir, img_name)
    if load_mode == 'gray':
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB)
    imgs.append(img)

# build model
c = 1 if load_mode == 'gray' else 3
if model_type == 'cnn':
    model = ResNet(c, c)
else:
    model = UNet(c, c)

train_name = '20250326_2053_gen_i2i_watermark_remove_train1.2_val1.2_image_gray_unet_20e_4bs'
weight_name = 'best.pt'
weight_path = f'runs/{train_name}/{weight_name}'
model.load_state_dict(torch.load(weight_path))

model.eval().to(device)

# predict
predictions = []
with torch.inference_mode():
    for i in tqdm(range(len(imgs))):
        img = imgs[i]
        img = transform(Image.fromarray(img))
        img = img.unsqueeze(0).to(device)
        cur_pred = model(img)
        cur_pred = cur_pred.cpu().numpy()
        #print(f'cur_pred shape: {cur_pred.shape}')
        predictions.append(cur_pred)

# to img
#out_img_dir = test_img_dir + '_pred'
out_img_dir = f'results/{train_name}_pred'
check_dir(out_img_dir)

for i, img in enumerate(imgs):
    if cat:
        img = imgs[i]
        img = np.resize(img, (resize_size, resize_size))
        h, w = img.shape[:2]
        gap_img = np.zeros((h, 16), dtype=np.uint8)
        pred = predictions[i][0]
        print(pred.shape)
        pred = np.transpose(pred, (1, 2, 0)) * 255
        pred = np.squeeze(pred)
        pred = pred.astype(np.uint8)
        print(f'img shape: {img.shape}, gap_img shape: {gap_img.shape}, pred shape: {pred.shape}')
        image = np.concatenate([img, gap_img, pred], axis=1)
    cv2.imwrite(os.path.join(out_img_dir, img_names[i]), image)
