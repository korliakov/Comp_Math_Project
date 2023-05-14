import numpy as np
import os
import subprocess
from tqdm.notebook import tqdm
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split
from torch.nn import BCEWithLogitsLoss
import matplotlib.pyplot as plt
from IPython.display import clear_output
from PIL import Image
import albumentations as A
from sklearn.cluster import DBSCAN


def conv_bn_relu(in_channels, out_channels, kernel=3, stride=1, padding=1):
    net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )
    return net


class Stacked2ConvsBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stacked2ConvsBlock, self).__init__()
        self.blocks = nn.Sequential(
            conv_bn_relu(in_channels, out_channels),
            conv_bn_relu(out_channels, out_channels)
        )

    def forward(self, net):
        net = self.blocks(net)
        return net


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSamplingBlock, self).__init__()

        # Понижаем число каналов
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2)

        # Стакаем с симметричным слоем из левой половины "U".
        # Число каналов входной карты при этом удваивается.
        self.convolve = Stacked2ConvsBlock(2 * in_channels, out_channels)

    def forward(self, left_net, right_net):
        right_net = self.upsample(right_net)
        # конкатенируем векторы из левой и правой половины
        net = torch.cat((left_net, right_net), dim=(1))  # <ваш код>
        # применяем свертку
        net = self.convolve(net)  # <ваш код>
        return net


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Stacked2ConvsBlock(in_channels, out_channels)
        )

    def forward(self, net):
        return self.blocks(net)


class Unet(nn.Module):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        super(Unet, self).__init__()
        self.init_conv = Stacked2ConvsBlock(1, 64)  ####

        self.downsample_1 = DownSamplingBlock(64, 128)
        self.downsample_2 = DownSamplingBlock(128, 256)
        self.downsample_3 = DownSamplingBlock(256, 512)
        self.downsample_4 = DownSamplingBlock(512, 1024)

        # В середине есть блок без пары с 1024 каналами
        # с ним конкатенировать некого, потому просто свернём его
        self.upconv = Stacked2ConvsBlock(1024, 512)

        # Подъём. Аналогично.
        self.upsample_1 = UpSamplingBlock(512, 256)
        self.upsample_2 = UpSamplingBlock(256, 128)
        self.upsample_3 = UpSamplingBlock(128, 64)
        # Чтобы учесть входной слой после самой первой свёртки и согласовать размерности
        self.upsample_4 = UpSamplingBlock(64, 64)

        # Предсказание
        self.agg_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        net0 = self.init_conv(x)  # 1 --> 64

        net1 = self.downsample_1(net0)  # 64 --> 128
        net2 = self.downsample_2(net1)  # 128 --> 256
        net3 = self.downsample_3(net2)  # 256 --> 512
        net = self.downsample_4(net3)  # 512 --> 1024

        net = self.upconv(net)  # 1024 --> 512

        net = self.upsample_1(net3, net)  # 512 --> 256
        net = self.upsample_2(net2, net)  # 256 --> 128
        net = self.upsample_3(net1, net)  # 128 --> 64
        net = self.upsample_4(net0, net)  # 64 --> 64

        net = self.agg_conv(net)  # 64 --> 1

        return net



def get_coords_from_mask(mask, xv, yv):
    return np.vstack((xv[mask > 0], yv[mask > 0])).T

def unet_make_prediction(image, model, threshold):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        pred = model(image.unsqueeze(dim=0).to(device)).squeeze().cpu()
    return pred > torch.logit(torch.FloatTensor([threshold]))


def init_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def dbscan_make_prediction(coords, eps, min_samples):

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    pred = dbscan.fit_predict(coords)
    return pred

