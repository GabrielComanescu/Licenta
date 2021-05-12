# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
import torchvision.utils as vutils
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
# UTILS

from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def train_hr_transform(crop_size):
    return transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.ToTensor()])

def train_lr_transform(crop_size, upscale_factor):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(crop_size // upscale_factor, interpolation = Image.BICUBIC),
        transforms.ToTensor()
    ])

def display_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(400),
        transforms.CenterCrop(400),
        transforms.ToTensor()
    ])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)
        
    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image
    
    def __len__(self):
        return len(self.image_filenames)
    
class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        
    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w,h),self.upscale_factor)
        lr_scale = transforms.Resize(crop_size // self.upscale_factor, interpolation = Image.BICUBIC)
        hr_scale = transforms.Resize(crop_size, interpolation = Image.BICUBIC)
        hr_image = transforms.CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return transforms.ToTensor()(lr_image), transforms.ToTensor()(hr_restore_img), transforms.ToTensor()(hr_image)
    
    def __len__(self):
        return len(self.image_filenames)
    
class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]
        
    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w,h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = transforms.Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, transforms.ToTensor()(lr_image), transforms.ToTensor()(hr_restore_img), transforms.ToTensor()(hr_image)
        


# %%
# LOSS

from torchvision.models.vgg import vgg19


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg19(pretrained = True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        
    def forward(self, out_labels, out_images, target_images):
        adversarial_loss = torch.mean(1 - out_labels)
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        image_loss = self.mse_loss(out_images, target_images)
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss
    
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight
        
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:,:,1:,:])
        count_w = self.tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:] - x[:,:,:h_x - 1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:] - x[:,:,:,:w_x - 1]),2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    
    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    
if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)


# %%
# RESIDUAL BLOCK

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels,channels, kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self,x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        
        return x + residual
    
    
    
# UPSAMPLE BLOCK

class UpsampleBlock(nn.Module):
    def __init__(self,in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        self.prelu(x)
        return x
    
    
    
# GENERATOR

import math


class Generator(nn.Module):
    
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor,2))
        
        super(Generator, self).__init__()
        
        self.block1 = nn.Sequential(nn.Conv2d(3,64,kernel_size=9,padding=4),nn.PReLU())
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,padding=1), nn.BatchNorm2d(64))
        block8 = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64,3,kernel_size=9,padding=4))
        self.block8 = nn.Sequential(*block8)
        
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)
        
        return (torch.tanh(block8) + 1) / 2


    
    
# DISCRIMINATOR

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.net = nn.Sequential(nn.Conv2d(3,64,kernel_size=3,padding=1),
                                 nn.LeakyReLU(0.2),
                                 
                                nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1),
                                nn.BatchNorm2d(64), 
                                nn.LeakyReLU(0.2),
                                
                                nn.Conv2d(64,128,kernel_size=3,padding=1),
                                nn.BatchNorm2d(128), 
                                nn.LeakyReLU(0.2),
                                 
                                nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1),
                                nn.BatchNorm2d(128), 
                                nn.LeakyReLU(0.2),
                                 
                                nn.Conv2d(128,256,kernel_size=3,padding=1),
                                nn.BatchNorm2d(256), 
                                nn.LeakyReLU(0.2),
                                 
                                nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1),
                                nn.BatchNorm2d(256), 
                                nn.LeakyReLU(0.2),
                                 
                                nn.Conv2d(256,512,kernel_size=3,padding=1),
                                nn.BatchNorm2d(512), 
                                nn.LeakyReLU(0.2),
                                 
                                nn.Conv2d(512,512,kernel_size=3,stride=2,padding=1),
                                nn.BatchNorm2d(512), 
                                nn.LeakyReLU(0.2),
                                 
                                nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(512,1024,kernel_size=1),
                                nn.LeakyReLU(0.2),
                                nn.Conv2d(1024,1,kernel_size=1)
                                )
        
    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


# %%
# netG = Generator(2).eval()
# netG.cuda()
# netG.load_state_dict(torch.load('modele/' + 'netG_epoch.pth'))

# netD = Discriminator().eval()
# netD.cuda()
# netD.load_state_dict(torch.load('modele/' + 'netD_epoch.pth'))

# generator_criterion = GeneratorLoss()
# generator_criterion.cuda()


# %%
# TEST

def shoot():
    import argparse
    import time

    import torch
    from PIL import Image
    from torch.autograd import Variable
    from torchvision.transforms import ToTensor, ToPILImage

    UPSCALE_FACTOR = 2
    IMAGE_NAME = 'pu.jpg'
    MODEL_NAME = 'netG_epoch.pth'

    model = Generator(UPSCALE_FACTOR).eval()
    model.cuda()
    model.load_state_dict(torch.load('modele/' + MODEL_NAME))

    image = Image.open(IMAGE_NAME)
    image = Variable(transforms.ToTensor()(image)).unsqueeze(0)
    image = image.cuda()

    with torch.no_grad():
        out = model(image)
        
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save('test/' + str('20') + '_' + IMAGE_NAME)

    print('succes conversie')
    torch.cuda.empty_cache()


