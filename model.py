import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms,utils,datasets
from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
from tqdm import *
import time
import random


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # prep Network:先对密图S 进行两层卷积，每次卷积完的特征图进行融合
        self.conv1 = nn.Conv2d(3, 50, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 10, 4, padding=1)
        self.conv3 = nn.Conv2d(3, 5, 5, padding=2)

        self.conv4 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv5 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv6 = nn.Conv2d(65, 5, 5, padding=2)

        # Hiding Network ：密图S在prep network 中的输出特征图，融合封面图片融合，并经过5层卷积
        # 原图3channel，密图65channel，故输入68个channel
        self.conv7 = nn.Conv2d(68, 50, 3, padding=1)
        self.conv8 = nn.Conv2d(68, 10, 4, padding=1)
        self.conv9 = nn.Conv2d(68, 5, 5, padding=2)

        self.conv10 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv11 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv12 = nn.Conv2d(65, 5, 5, padding=2)

        self.conv13 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv14 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv15 = nn.Conv2d(65, 5, 5, padding=2)

        self.conv16 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv17 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv18 = nn.Conv2d(65, 5, 5, padding=2)

        self.conv19 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv20 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv21 = nn.Conv2d(65, 5, 5, padding=2)

        self.conv22 = nn.Conv2d(65, 3, 3, padding=1)

    def forward(self, input_S, input_C):
        x1 = F.relu(self.conv1(input_S))
        x2 = F.relu(self.conv2(input_S))
        # kernel_size=4,卷积之后，高和宽的单位减 3 ，padding=1,之后，
        # 四周加一条边，高和宽的单位加 2 ，但高和宽较原图任然缺 1 个单位,（0，1，0，1）表示在图像右下角加横纵两条边
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv3(input_S))
        x4 = torch.cat((x1, x2, x3), 1)  # 特征图变厚,

        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv6(x4))
        x4 = torch.cat((x1, x2, x3), 1)

        x4 = torch.cat((input_C, x4), 1)

        # 进入hiding network
        x1 = F.relu(self.conv7(x4))
        x2 = F.relu(self.conv8(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv9(x4))
        x4 = torch.cat((x1, x2, x3), 1)

        x1 = F.relu(self.conv10(x4))
        x2 = F.relu(self.conv11(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv12(x4))
        x4 = torch.cat((x1, x2, x3), 1)

        x1 = F.relu(self.conv13(x4))
        x2 = F.relu(self.conv14(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv15(x4))
        x4 = torch.cat((x1, x2, x3), 1)

        x1 = F.relu(self.conv16(x4))
        x2 = F.relu(self.conv17(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv18(x4))
        x4 = torch.cat((x1, x2, x3), 1)

        x1 = F.relu(self.conv19(x4))
        x2 = F.relu(self.conv20(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv21(x4))
        x4 = torch.cat((x1, x2, x3), 1)

        # 再进行一层卷积,输出3个channel（RBG）
        output = torch.tanh(self.conv22(x4))

        return output


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # 解码器，重建密图
        self.conv1 = nn.Conv2d(3, 50, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 10, 4, padding=1)
        self.conv3 = nn.Conv2d(3, 5, 5, padding=2)

        self.conv4 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv5 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv6 = nn.Conv2d(65, 5, 5, padding=2)

        self.conv7 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv8 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv9 = nn.Conv2d(65, 5, 5, padding=2)

        self.conv10 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv11 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv12 = nn.Conv2d(65, 5, 5, padding=2)

        self.conv13 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv14 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv15 = nn.Conv2d(65, 5, 5, padding=2)

        self.conv16 = nn.Conv2d(65, 3, 3, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv3(x))
        x4 = torch.cat((x1, x2, x3), 1)

        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv6(x4))
        x4 = torch.cat((x1, x2, x3), 1)

        x1 = F.relu(self.conv7(x4))
        x2 = F.relu(self.conv8(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv9(x4))
        x4 = torch.cat((x1, x2, x3), 1)

        x1 = F.relu(self.conv10(x4))
        x2 = F.relu(self.conv11(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv12(x4))
        x4 = torch.cat((x1, x2, x3), 1)

        x1 = F.relu(self.conv13(x4))
        x2 = F.relu(self.conv14(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv15(x4))
        x4 = torch.cat((x1, x2, x3), 1)

        output = torch.tanh(self.conv16(x4))

        return output


class Make_model(nn.Module):
    def __init__(self):
        super(Make_model, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input_S, input_C):
        output_Cprime = self.encoder(input_S, input_C)
        output_Sprime = self.decoder(output_Cprime)

        return output_Cprime, output_Sprime