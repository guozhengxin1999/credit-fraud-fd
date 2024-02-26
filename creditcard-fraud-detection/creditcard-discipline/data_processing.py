import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
import draw
import math
import random


local_epochs = 20  # 选用20个epoch
loss_func = nn.MSELoss()  # 采用均方差损失函数



# 定义自编码器网络结构，这里采用最简单的自动编码器。
encoding_dim_s = 12
encoding_dim = 16
encoding_dim1= 64
input_dim = x_train.shape[1]
threshhold_num = 1.6

class AutoEncoder_Teacher(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder_Teacher, self).__init__()

        ## 定义编码器结构
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, encoding_dim1),
            torch.nn.ReLU(),
            torch.nn.Linear(encoding_dim1,encoding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(encoding_dim, 4),
            torch.nn.ReLU())

        ## 定义解码器结构
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4, encoding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(encoding_dim, encoding_dim1),
            torch.nn.ReLU(),
            torch.nn.Linear(encoding_dim1, input_dim)
        )

    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output


def teacher_main():
    model = AutoEncoder_Teacher()
    # 创建分类器
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    teacher_history = []
    acc_history = []
    for epoch in range(1, local_epochs + 1):
        total_loss = train_teacher(model, optimizer, epoch)
        total_loss /= len(train_set)
        teacher_history.append(total_loss)

        recon_err_test = torch.mean((model(x_test) - x_test) ** 2, dim=1).detach().numpy()
        y_pred = (recon_err_test > threshhold_num).astype(np.int)
        acc_history .append(accuracy_score(y_pred, y_test))

        print('Teacher Epoch {}/{} : loss: {:.4f}, acc:{:.4f}'.format(
            epoch , local_epochs , total_loss, accuracy_score(y_pred, y_test)))
        # loss, acc = test_teacher(model)

        # teacher_history.append((loss, acc))

    torch.save(model.state_dict(), "../teacher.pt")
    # print(teacher_history)
    return model, teacher_history, acc_history



def train_teacher(model,optimizer, epoch):
    model.train()
    trained_samples = 0
    total_loss = 0

    # 训练参数
    for step, (data, ) in enumerate(train_loader):
        # 清空过往梯度
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, data)
        # 反向传播，计算当前梯度
        loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()

        total_loss += loss.item() * len(data)
        trained_samples += len(data)
        progress = math.ceil(step / len(train_loader) * 50)

    return total_loss
