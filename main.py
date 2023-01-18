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

from data_process import *
from model import *
from test  import test


# 训练

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # torch.cuda.set_device(1) #选择第二块显卡
    print("device : ", device)

    AEmodel.to(device)

    train_loader, test_loader, train_set_size, test_set_size = get_sub_data_loaders()

    # 定义损失函数
    S_mseloss = torch.nn.MSELoss().to(device)  # 计算密图重建损失
    C_mseloss = torch.nn.MSELoss().to(device)  # 计算明图重建损失

    optimizer = torch.optim.Adam(AEmodel.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9)
    loss_func = nn.CrossEntropyLoss().cuda()  # the target label is not one-hotted

    for epoch in range(100):

        loss_all, c_loss, s_loss = [], [], []
        t = tqdm(train_loader)  # 创建进度条,可直接用在迭代对象data_loader上

        for images, _ in t:
            images = images.to(device)  # 数据发送到GPU，不然数据在cup上，格式不同
            AEmodel.train()  # 标记状态

            with torch.enable_grad():
                optimizer.zero_grad()  # 梯度清零

                # images 为一个batch的图像，分为明图和密图两部分
                input_C = images[0:images.shape[0] // 2]
                input_S = images[images.shape[0] // 2:]

                output_C, output_S = AEmodel(input_S, input_C)

                # 计算损失
                beta = 1.0
                ssLoss = S_mseloss(input_S, output_S)
                ccLoss = C_mseloss(input_C, output_C)
                loss = beta * ssLoss + ccLoss

                loss.backward()
                optimizer.step()

                losses = {
                    "loss_all": loss.item(),
                    "ssLoss": ssLoss.item(),
                    "ccLoss": ccLoss.item()
                }
                loss_all.append(losses["loss_all"])
                c_loss.append(losses["ccLoss"])
                s_loss.append(losses["ssLoss"])

        loss_history.append(loss_all)
        print("[epoch = ", epoch + 1, "] loss: ", np.mean(loss_all), "s_loss = ", np.mean(c_loss), "c_loss = ",
              np.mean(s_loss))


if __name__ == '__main__':
    # 随机显示一些图像
    fig = plt.figure(figsize=(8, 8))  # 图像尺寸
    colums = 4
    rows = 5

    # 获取数据
    train_loader, test_loader, train_set_size, test_set_size = get_sub_data_loaders()

    print("train_images len = ", train_set_size)
    print("test_images len  = ", test_set_size)

    # 随机获取一个batch的图像
    dataiter = iter(train_loader)
    # images,labels = dataiter.next()
    images, labels = next(dataiter)
    # print("images ",images)
    print("labels ", labels)
    print("images size = ", images.size())

    for i in range(1, colums * rows + 1):
        fig.add_subplot(rows, colums, i)
        # torch image 格式转换为 numpy image
        img = images[i] / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 维度转换
    plt.show()

    AEmodel = Make_model()
    loss_history = []

    train()

    plt.plot(loss_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    # save model
    path = "./modelgpu_3200.pth"
    torch.save(AEmodel.state_dict(), path)

    test()






