import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml

from common_utils import check_dir, get_homedir

from dataset.denoise_dataset import DenoisePatchDataset, DenoiseImageDataset
from dataset.data_aug import get_transform
from models.cnn import CNN, ResNet
from models.unet import UNet
from utils import get_train_name, log_configs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='', help='train configure file')

    return parser.parse_args()


def train(configs, model, device, train_loader, loss_fn, optimizer, epoch, logfile, writer):
    model.train().to(device)
    clipnorm = configs['clipnorm']

    for batch_idx, batch in enumerate(train_loader):
        images = batch['image']
        labels = batch['label']
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
        optimizer.step()

        step = epoch * len(train_loader) + batch_idx

        if batch_idx % configs['print_interval'] == 0:
            cur_idx = batch_idx * len(images)
            num_patch = len(train_loader.dataset)
            logfile.write(f'{epoch:05d} Epoch, [{cur_idx}/{num_patch} ({cur_idx/num_patch*100:.0f}%)]\tLoss: {loss.item():.6f}\n')
            logfile.flush()
            writer.add_scalar('Loss/train_loss', loss.item(), step)


def validate(model, device, val_loader, loss_fn, logfile):
    model.eval().to(device)

    val_loss = 0
    with torch.inference_mode():
        for batch in val_loader:
            images = batch['image']
            labels = batch['label']
            images, labels = images.to(device), labels.to(device)
            preds = model(images)

            val_loss += loss_fn(preds, labels)

    val_loss = val_loss / len(val_loader)

    logfile.write(f'\nAverage val loss: {val_loss:.6f}\n')
    logfile.flush()

    return val_loss


def main(args):
    config_path = args.config
    assert config_path != '', print('need config file!')

    with open(config_path, 'r') as f:
        yaml_content = f.read()
    configs = yaml.safe_load(yaml_content)

    homedir = get_homedir()
    # load data
    BATCHSIZE = configs['batch_size']
    val_batch_size = configs['val_batch_size']
    train_data_list = os.path.join(homedir, configs['train_data_list'])
    val_data_list = os.path.join(homedir, configs['val_data_list'])
    dataloader_num_workers = configs['dataloader_num_workers']
    patch_size = configs['patch_size']
    overlap = configs['overlap']
    load_mode = configs['load_mode']
    transform = get_transform(configs)
    assert load_mode in ['gray', 'rgb'], print(f'load_mode should be gray or rgb')
    data_mode = configs['data_mode']
    assert data_mode in ['patch', 'image'], print(f'data_mode should be patch or image')
    if data_mode == 'patch':
        train_set = DenoisePatchDataset(train_data_list, load_mode, transform=transform, patch_size=patch_size, overlap=overlap)
    else:
        train_set = DenoiseImageDataset(train_data_list, load_mode=load_mode, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True, drop_last=True, num_workers=dataloader_num_workers)
    if data_mode == 'patch':
        val_set = DenoisePatchDataset(val_data_list, load_mode, transform=transform, patch_size=patch_size, overlap=overlap)
    else:
        val_set = DenoiseImageDataset(val_data_list, load_mode=load_mode, transform=transform)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=True, drop_last=True, num_workers=dataloader_num_workers)

    # build model
    model_type = configs['model_type']
    assert model_type in ['cnn', 'resnet', 'unet'], print('model_type must be cnn, resnet or unet')
    c = 1 if load_mode == 'gray' else 3
    if model_type == 'cnn':
        model = CNN(c, c)
    elif model_type == 'resnet':
        model = ResNet(c, c)
    else:
        model = UNet(c, c)

    # loss and optimizer
    EPOCHS = configs['epochs']
    lr = configs['lr']
    beta_1 = configs['beta_1']
    beta_2 = configs['beta_2']
    weight_decay = configs['weight_decay']
    clipnorm = configs['clipnorm']
    #loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    #val_loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta_1, beta_2), weight_decay=weight_decay)

    # train
    device = configs['device']
    val_device = configs['val_device']
    logs_dir = configs['logs_dir']
    save_root_dir = configs['save_root_dir']

    train_name = get_train_name(configs)
    logfile_path = os.path.join(logs_dir, f'{train_name}.log')
    save_dir = os.path.join(save_root_dir, train_name)

    if os.path.exists(logfile_path):
        os.remove(logfile_path)
    logfile = open(logfile_path, 'a')
    log_configs(logfile, configs)

    check_dir(save_dir)
    writer = SummaryWriter(save_dir)

    best_val_loss = float('inf')
    best_weight_name = 'best.pt'
    last_weight_name = 'last.pt'
    best_weight_path = os.path.join(save_dir, best_weight_name)
    last_weight_path = os.path.join(save_dir, last_weight_name)
    for epoch in range(1, EPOCHS + 1):
        train(configs, model, device, train_loader, loss_fn, optimizer, epoch, logfile, writer)
        val_loss = validate(model, val_device, val_loader, loss_fn, logfile)

        step = epoch * len(train_loader)
        writer.add_scalar('Loss/val_loss', val_loss, step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weight_path)

        epoch_weight_name = f'{epoch}epoch.pt'
        epoch_weight_path = os.path.join(save_dir, epoch_weight_name)
        torch.save(model.state_dict(), epoch_weight_path)

    torch.save(model.state_dict(), last_weight_path)

    logfile.write(f'best val loss: {best_val_loss:.6f}\n')
    logfile.close()
    writer.close()


if __name__ == '__main__':
    args = get_args()
    main(args)
