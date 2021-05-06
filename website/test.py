from loss import GeneratorLoss
from model import Generator, Discriminator
import os

# TEST
import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage


# netG = Generator(2).eval()
# netG.cuda()
# netG.load_state_dict(torch.load('modele/netG_epoch.pth'))

# netD = Discriminator().eval()
# netD.cuda()
# netD.load_state_dict(torch.load('modele/netD_epoch.pth'))

# generator_criterion = GeneratorLoss()
# generator_criterion.cuda()


def shoot():
    torch.cuda.empty_cache()
    UPSCALE_FACTOR = 2
    TEST_MODE = True
    IMAGE_NAME = 'pu.jpg'
    MODEL_NAME = 'netG_epoch.pth'

    model = Generator(UPSCALE_FACTOR).eval()
    model.cuda()
    model.load_state_dict(torch.load('modele/' + MODEL_NAME))

    image = Image.open(IMAGE_NAME)
    image = Variable(ToTensor()(image)).unsqueeze(0)
    image = image.cuda()

    with torch.no_grad():
        out = model(image)
        
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save('test/' + str('20') + '_' + IMAGE_NAME)