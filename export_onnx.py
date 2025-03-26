import os
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import onnxruntime as ort
from tqdm import tqdm
import time

from common_utils import norm_homedir
from draw_utils import show_tmp

from models.cnn import CNN, ResNet 
from utils import img2patch, patch2img


def export():
    # load model
    model = ResNet()
    weight_dir = 'runs/20250319_1724_gen_i2i_watermark_remove_train0.0_val0.0_20e_20bs'
    weight_path = os.path.join(weight_dir, 'best.pt')
    model.load_state_dict(torch.load(weight_path))

    model.eval()

    # export onnx format
    onnx_weight_name = 'best.onnx'
    onnx_weight_path = os.path.join(weight_dir, onnx_weight_name)
    #dummy_input = torch.randint(low=0, high=256, size=(1, 1, 50, 50), dtype=torch.uint8)
    dummy_input = torch.randn((1, 1, 50, 50))
    torch.onnx.export(
        model, dummy_input, onnx_weight_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )


# onnx inference
def onnx_inference_one(img_path):
    patch_size = 50
    overlap = 30

    st = time.time()
    weight_dir = 'runs/20250319_1724_gen_i2i_watermark_remove_train0.0_val0.0_20e_20bs'
    onnx_weight_path = os.path.join(weight_dir, 'best.onnx')
    ort_session = ort.InferenceSession(onnx_weight_path)
    et = time.time()
    print(f'start ort_session cost: {et-st:.2f}s')
    img = cv2.imread(img_path, 0)
    print(img.shape)
    data = np.array(img2patch([img], patch_size, overlap), dtype=np.float32)
    data = data.reshape(data.shape[0], 1, 50, 50)
    print(data.shape)

    outputs = ort_session.run(None, {'input': data})
    print(f'outputs len:{len(outputs)}')
    print(f'outputs[0] type:{type(outputs[0])}')
    print(f'outputs[0] shape:{outputs[0].shape}')

    images = patch2img(outputs[0], [img], patch_size, overlap)
    image = images[0]
    show_tmp(image)


if __name__ == '__main__':
    #export()

    img_path = '/data4/hanyp/watermark/denoise_watermark/dataset/zyb/zyb_0c551f4a5827d473c13cbd6e53e141b7.jpg'
    onnx_inference_one(img_path)
