import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split

df = pd.read_csv("creditcard-discipline/creditcard.csv")
data = df.iloc[:, 1:31]

# 样本不平衡，需要采用下采样策略，减小多数类使其数量与少数类相同
X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']

number_records_fraud = len(data[data.Class == 1])  # class=1的样本函数
fraud_indices = np.array(data[data.Class == 1].index)  # 样本等于1的索引值

normal_indices = data[data.Class == 0].index  # 样本等于0的索引值

random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)

under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])  # Appending the 2 indices

under_sample_data = data.iloc[under_sample_indices, :]  # Under sample dataset

X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class']

X_train, X_test, y_train, y_test = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0)

X_train = torch.from_numpy(X_train.values)
X_test = torch.from_numpy(X_test.values)
y_train = torch.from_numpy(y_train.values)
y_test = torch.from_numpy(y_test.values)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义三层线性层  通过矩阵乘法将前一层的矩阵变换为下一层的矩阵
        self.linear1 = nn.Linear(29, 20)  # 759x8 -> 759x6
        self.linear2 = nn.Linear(20, 11)  # 759x6 -> 759x4
        self.linear3 = nn.Linear(11, 1)  # 759x4 -> 759x1
        # 定义激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 叠加线性层每两层之间一定要加入非线性层，否则可以直接由一个线性层代替
        x = self.sigmoid(self.linear1(x))  # 759x8
        x = self.sigmoid(self.linear2(x))  # 759x6
        y_pred = self.sigmoid(self.linear3(x))  # 759x4
        return y_pred  # 759x1

def train():
    model = Model()

    ## 损失函数和优化器
    # torch.nn.BCELoss() 二分类交叉熵损失函数（ Binary CrossEntropy ）
    # reduction = 'mean' ，返回loss的平均值
    criterion = nn.BCELoss(reduction='mean')

    # torch.optim.SGD() 随机梯度下降算法（ Stochastic Gradient Descent ）
    # model.parameters() 保存的是Weights和Bais参数的值 y=wx+b
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


    loss_list = []
    epoch_list = []

    for epoch in range(10):
        y_pred = model(X_train.to(torch.float32))
        loss = criterion(y_pred, y_test.to(torch.float32))
        loss_list.append(loss.item())
        epoch_list.append(epoch)
        # 清空过往梯度
        optimizer.zero_grad()
        # 反向传播，计算当前梯度
        loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()
        print(epoch, loss.item())

    plt.plot(epoch_list, loss_list)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


if __name__ == '__main__':
    train()
