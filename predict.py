import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm

from common_utils import check_dir

cur_path = os.path.abspath(__file__)
proj_root_path = os.path.dirname(cur_path)
sys.path.append(proj_root_path)
from models.cnn import CNN, ResNet 
from models.unet import UNet
from utils import load_imgs, img2patch, patch2img


patch_size = 50
overlap = 30
device = 'cuda:3'
bs = 32
cat = True
load_mode = 'gray'
model_type = 'unet'
resize_size = 60
crop_size = 50

transform = transforms.ToTensor()

# load imgs
#test_img_dir = '/data4/hanyp/watermark/denoise_watermark/dataset/zyb_imgs_300_val'
#test_img_dir = '/data4/hanyp/watermark/denoise_watermark/dataset/zyb'
test_img_dir = 'test_case'
ori_imgs, names = load_imgs(test_img_dir)

data = np.array(img2patch(ori_imgs, patch_size, overlap))
data = data.reshape(data.shape[0], 1, 50, 50)
data = torch.tensor(data, dtype=torch.float).to(device)

# build model
#model = ResNet()
model = UNet()

train_name = '20250326_1252_gen_i2i_watermark_remove_train1.2_val1.2_20e_20bs'
weight_name = 'best.pt'
weight_path = f'runs/{train_name}/{weight_name}'
model.load_state_dict(torch.load(weight_path))

model.eval().to(device)

# predict
predictions = []
with torch.inference_mode():
    for i in tqdm(range(0, len(data), bs)):
        cur_data = data[i: min(i+bs, len(data))]
        cur_pred = model(cur_data)
        cur_pred = cur_pred.cpu().numpy()
        #print(f'cur_pred shape: {cur_pred.shape}')
        predictions.append(cur_pred)
predictions = np.concatenate(predictions, axis=0)
print(predictions.shape)
#predictions = predictions.cpu().numpy()

# to img
#out_img_dir = test_img_dir + '_pred'
out_img_dir = f'results/{train_name}_pred_8e'
check_dir(out_img_dir)
images = patch2img(predictions, ori_imgs, patch_size=patch_size, overlap=overlap)

for i, image in enumerate(images):
    if cat:
        ori_img = ori_imgs[i]
        h, w = ori_img.shape
        gap_img = np.zeros((h, 16), dtype=np.uint8)
        image = np.concatenate([ori_img, gap_img, image], axis=1)
    cv2.imwrite(os.path.join(out_img_dir, names[i]), image)
