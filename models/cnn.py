import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, ic, oc):
        super(CNN, self).__init__()
        
        # hidden units
        nstates = [96, 96, 96, 96, 96]
        
        # model layers
        self.conv1 = nn.Conv2d(ic, nstates[0], kernel_size=3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.1)
        
        self.conv2 = nn.Conv2d(nstates[0], nstates[1], kernel_size=1)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.1)
        
        self.conv3 = nn.Conv2d(nstates[1], nstates[2], kernel_size=3, stride=1, padding=1)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.1)
        
        self.conv4 = nn.Conv2d(nstates[2], nstates[3], kernel_size=1)
        self.lrelu4 = nn.LeakyReLU(negative_slope=0.1)
        
        self.conv5 = nn.Conv2d(nstates[3], nstates[4], kernel_size=3, stride=1, padding=1)
        self.lrelu5 = nn.LeakyReLU(negative_slope=0.1)
        
        self.conv6 = nn.Conv2d(nstates[4], oc, kernel_size=3, stride=1, padding=1)
        self.lrelu6 = nn.LeakyReLU(negative_slope=0.001)
        self.lambda1 = lambda x: -x + 1
        self.lrelu7 = nn.LeakyReLU(negative_slope=0.001)
        self.lambda2 = lambda x: -x + 1
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        
        x = self.conv2(x)
        x = self.lrelu2(x)
        
        x = self.conv3(x)
        x = self.lrelu3(x)
        
        x = self.conv4(x)
        x = self.lrelu4(x)
        
        x = self.conv5(x)
        x = self.lrelu5(x)
        
        x = self.conv6(x)
        x = self.lrelu6(x)
        x = self.lambda1(x)
        x = self.lrelu7(x)
        x = self.lambda2(x)
        
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.1)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.lrelu1(out)
        out = self.conv2(out)
        out = self.lrelu2(out)
        out += identity
        return out


class ResNet(nn.Module):
    def __init__(self, ic, oc):
        super(ResNet, self).__init__()

        # hidden units
        nstates = [96, 96, 96, 96, 96]

        # model layers
        self.conv1 = nn.Conv2d(ic, nstates[0], kernel_size=3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.1)

        self.res_block1 = ResidualBlock(nstates[0], nstates[1], kernel_size=1, stride=1, padding=0)
        self.res_block2 = ResidualBlock(nstates[1], nstates[2], kernel_size=3, stride=1, padding=1)
        self.res_block3 = ResidualBlock(nstates[2], nstates[3], kernel_size=1, stride=1, padding=0)
        self.res_block4 = ResidualBlock(nstates[3], nstates[4], kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(nstates[4], oc, kernel_size=3, stride=1, padding=1)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.001)
        self.lambda1 = lambda x: -x + 1
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.001)
        self.lambda2 = lambda x: -x + 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.lambda1(x)
        x = self.lrelu3(x)
        x = self.lambda2(x)

        return x


if __name__ == '__main__':
    # 示例使用
    model = DenoiseModel()
    
    # 打印模型架构
    print(model)

    import os
    import sys
    from torch.utils.data import DataLoader

    cur_path = os.path.abspath(__file__)
    proj_root_path = os.path.dirname(os.path.dirname(cur_path))
    sys.path.append(proj_root_path)
    from dataset.denoise_dataset import DenoiseDataset

    ori_img_dir = '/data4/hanyp/watermark/denoise_watermark/dataset/train'
    cleaned_img_dir = '/data4/hanyp/watermark/denoise_watermark/dataset/train_cleaned'

    train_set = DenoiseDataset(ori_img_dir, cleaned_img_dir)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, drop_last=True)
    print(f'train_loader len: {len(train_loader)}')
    print()
    print(len(train_set))
    for ori_data, cleaned_data in train_loader:
        ori_data = ori_data.float()
        cleaned_data = cleaned_data.float()
        print(f'ori_data shape: {ori_data.shape}')
        print(f'cleaned_data shape: {cleaned_data .shape}')
        preds = model(ori_data)
        print(f'preds shape: {preds.shape}')
        break
