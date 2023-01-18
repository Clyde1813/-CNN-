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
from model import *

path = "./modelgpu_3200.pth"

def test():
    AEmodel = Make_model()
    AEmodel.load_state_dict(torch.load(path))
    AEmodel.eval()

    # 随机显示一些图像
    fig = plt.figure(figsize=(8, 8))  # 图像尺寸
    colums = 4
    rows = 5

    # 随机获取一个batch的图像
    dataiter = iter(test_loader)
    # images,labels = dataiter.next()
    images, labels = next(dataiter)
    # print("images ",images)
    print("labels ", labels)
    print("images size = ", images.size())

    # 将一个batch 一分为二，分为 C 与 S

    input_C = images[0:5]
    input_S = images[5:10]
    print("images c size = ", input_C.size())
    print("images s size = ", input_S.size())

    for i in range(0, rows):
        fig.add_subplot(rows, colums, i * colums + 1)

        img = input_C[i] / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        # print('npimg:',npimg)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 维度转换

        fig.add_subplot(rows, colums, i * colums + 1 + 1)
        s_img = input_S[i] / 2 + 0.5
        npimg_s = s_img.numpy()
        # print('npimg_s:',npimg_s)
        plt.imshow(np.transpose(npimg_s, (1, 2, 0)))  # 维度转换

        c_img_encode, s_img_decode = AEmodel(input_S[i].unsqueeze(0), input_C[i].unsqueeze(0))  # 获取编码图和密图

        fig.add_subplot(rows, colums, i * colums + 2 + 1)
        c_img_encode = c_img_encode / 2 + 0.5
        npc_img_encode = c_img_encode.detach().squeeze().numpy()
        # print('npc_img_encode:',npc_img_encode)
        plt.imshow(np.transpose(npc_img_encode, (1, 2, 0)))  # 维度转换

        fig.add_subplot(rows, colums, i * colums + 3 + 1)
        s_img_decode = s_img_decode / 2 + 0.5
        nps_img_decode = s_img_decode.detach().squeeze().numpy()
        # print('nps_img_decode:',nps_img_decode)
        plt.imshow(np.transpose(nps_img_decode, (1, 2, 0)))  # 维度转换

    plt.show()


