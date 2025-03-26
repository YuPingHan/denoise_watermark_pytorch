import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.upconv4 = self.upconv(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)

        self.upconv3 = self.upconv(512, 256)
        self.decoder3 = self.conv_block(512, 256)

        self.upconv2 = self.upconv(256, 128)
        self.decoder2 = self.conv_block(256, 128)

        self.upconv1 = self.upconv(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2, stride=2))

        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2, stride=2))

        dec4 = self.upconv4(bottleneck)
        dec4 = F.pad(dec4, [0, enc4.size(3) - dec4.size(3), 0, enc4.size(2) - dec4.size(2)])
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = F.pad(dec3, [0, enc3.size(3) - dec3.size(3), 0, enc3.size(2) - dec3.size(2)])
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = F.pad(dec2, [0, enc2.size(3) - dec2.size(3), 0, enc2.size(2) - dec2.size(2)])
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = F.pad(dec1, [0, enc1.size(3) - dec1.size(3), 0, enc1.size(2) - dec1.size(2)])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv_last(dec1)


if __name__ == '__main__':
    # 示例使用
    model = UNet()
    #print(model)

    import os
    import sys
    from torch.utils.data import DataLoader

    cur_path = os.path.abspath(__file__)
    proj_root_path = os.path.dirname(os.path.dirname(cur_path))
    sys.path.append(proj_root_path)
    from dataset.denoise_dataset import DenoiseDataset

    ori_img_dir = '/data4/hanyp/dataset/gen_i2i_watermark_remove/finals/gen_i2i_watermark_remove_val_v1.2_images'
    cleaned_img_dir = '/data4/hanyp/dataset/gen_i2i_watermark_remove/finals/gen_i2i_watermark_remove_val_v1.2_cleaned_images'
    data_list = '/data4/hanyp/dataset/gen_i2i_watermark_remove/finals/gen_i2i_watermark_remove_val_v1.2.txt'

    train_set = DenoiseDataset(data_list)
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
