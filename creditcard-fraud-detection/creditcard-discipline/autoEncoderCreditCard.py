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
plt.rcParams['font.family'] = 'Times New Roman'

local_epochs = 20  # 选用20个epoch
loss_func = nn.MSELoss()  # 采用均方差损失函数

# 载入数据并对数据预处理
# 数据共有280000多行，其中正常数据有280000行，异常数据共有500个左右，
# 选择40000个正常数据集和所有异常数据集，并将Amount列数据标准化。
df = pd.read_csv(r'creditcard.csv')
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
features = df.columns[:-1].tolist()
df0 = df.query('Class==0').sample(40000)
df1= df.query('Class==1')
x_nom = df0
x_nov = df1


# 划分训练集和测试集
# 划分数据集将85%的正常数据集作为训练集，将15%的正常数据集和异常数据集作为测试集
x_train, x_nom_test = train_test_split(x_nom.drop(labels=['Time','Class'],axis=1), train_size = 0.85, random_state = 1)
x_test = np.concatenate([x_nom_test,x_nov.drop(labels=['Time','Class'],axis=1)],axis = 0)
y_test = np.concatenate([np.zeros(len(x_nom_test)),np.ones(len(x_nov))])
# 处理数据格式为张量，并将其读入加载器
# 做数据格式的改变到张量
x_train = np.array(x_train)
x_test = np.array(x_test)

x_train, x_test = torch.FloatTensor(x_train), torch.FloatTensor(x_test)

# 将训练数据处理为数据加载器
train_set = TensorDataset(x_train)
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_set = TensorDataset(x_test)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True)


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


class AutoEncoder_Student(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder_Student, self).__init__()

        ## 定义编码器结构
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, encoding_dim_s),
            torch.nn.ReLU(),
            torch.nn.Linear(encoding_dim_s, 4),
            torch.nn.ReLU())

        ## 定义解码器结构
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4, encoding_dim_s),
            torch.nn.ReLU(),
            torch.nn.Linear(encoding_dim_s, input_dim)
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

def student_main():
    model = AutoEncoder_Student()
    # 创建分类器
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    student_history = []
    acc_history = []
    for epoch in range(1, local_epochs + 1):
        total_loss = train_student(model, optimizer, epoch)
        total_loss /= len(train_set)
        student_history.append(total_loss)

        recon_err_test = torch.mean((model(x_test) - x_test) ** 2, dim=1).detach().numpy()
        y_pred = (recon_err_test > threshhold_num).astype(np.int)
        acc_history.append(accuracy_score(y_pred, y_test))

        print('Student Epoch {}/{} : loss: {:.4f}, acc:{:.4f}'.format(
                 epoch , local_epochs, total_loss,accuracy_score(y_pred, y_test)))
        # loss, acc = test_teacher(model)

        # teacher_history.append((loss, acc))

    torch.save(model.state_dict(), "../student.pt")
    return model, student_history,acc_history


def train_student(model, optimizer, epoch):
    model.train()
    trained_samples = 0
    total_loss = 0

    # 训练参数
    for step, (data,) in enumerate(train_loader):
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
        # print("\rStudent Train epoch %d:%d%d,[%-51s]%d%%" %
        #       (epoch, trained_samples, len(train_loader.dataset),
        #        "-" * progress + '>', progress * 2), end='')

    return total_loss



#蒸馏模型的损失函数
def distillation(student_output, labels, teacher_scores, temp, alpha):
    distillation_loss = loss_func(F.softmax(student_output/temp, dim = 1),F.softmax(teacher_scores/temp, dim = 1))
    return alpha*loss_func(student_output,labels)+(1-alpha)*distillation_loss
    # return nn.MSELoss()(student_output / temp, teacher_scores / temp) * (
    #         temp * temp * 2.0 * alpha) + loss_func(student_output, labels) * (1. - alpha)

def student_kd_main():
    model = AutoEncoder_Student()
    # 创建分类器
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    student_kd_history = []
    acc_history = []
    for epoch in range(1, local_epochs + 1):
        model ,total_loss = train_student_kd(model, optimizer, epoch)
        total_loss /= len(train_set)
        student_kd_history.append(total_loss)
        with torch.no_grad():
            recon_err_test = torch.mean((model(x_test) - x_test) ** 2, dim=1).detach().numpy()
            y_pred = (recon_err_test > 6).astype(np.int)
            acc_history.append(accuracy_score(y_pred, y_test))

        print('Student KD Epoch {}/{} : loss: {:.4f}, acc:{:.4f}'.format(
            epoch, local_epochs, total_loss, accuracy_score(y_pred, y_test)))
        # loss, acc = test_teacher(model)

        # teacher_history.append((loss, acc))

    torch.save(model.state_dict(), "../student_kd.pt")
    print(student_kd_history)
    return model, student_kd_history,acc_history


def train_student_kd(model, optimizer, epoch):
    model.train()
    trained_samples = 0
    total_loss = 0

    # 训练参数
    for step, (data,) in enumerate(train_loader):
        # 清空过往梯度
        optimizer.zero_grad()
        with torch.no_grad(): # 教师网络不用反向传播
            teacher_output = teacher_model(data)

        student_output = model(data)
        loss = distillation(student_output, data, teacher_output, 4.0, 0.7)
        # 反向传播，计算当前梯度
        loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()

        total_loss += loss.item() * len(data)
        trained_samples += len(data)

    return model,total_loss


def get_recon_err(X):
    return torch.mean((teacher_model(X) - X) ** 2, dim=1).detach().numpy()


def err():
    recon_err_train = get_recon_err(x_train)
    recon_err_test = get_recon_err(x_test)
    recon_err = np.concatenate([recon_err_train, recon_err_test])
    labels = np.concatenate([np.zeros(len(recon_err_train)), y_test])
    index = np.arange(0, len(labels))

    sns.kdeplot(recon_err[labels == 0], shade=True)
    sns.kdeplot(recon_err[labels == 1], shade=True)
    plt.show()
    return recon_err_test

def threshhold(y_test, recon_err_test):
    threshold = np.linspace(0, 10, 200)
    acc_list = []
    f1_list = []

    for t in threshold:
        y_pred = (recon_err_test > t).astype(np.int)
        acc_list.append(accuracy_score(y_pred, y_test))
        f1_list.append(f1_score(y_pred, y_test))

    plt.figure(figsize=(8, 6))
    plt.plot(threshold, acc_list, c='y', label='acc')
    plt.plot(threshold, f1_list, c='b', label='f1')
    plt.xlabel('threshold')
    plt.ylabel('classification score')
    plt.legend()
    plt.show()

    i = np.argmax(f1_list)
    t = threshold[i]
    score = f1_list[i]
    print('threshold: %.3f,  f1 score: %.3f' % (t, score))

    viz = draw.visualization()
    viz.draw_confusion_matrix(y_test, y_pred)
    viz.draw_anomaly(y_test, recon_err_test, threshold[i])
    c = list(zip(y_test, y_pred, recon_err_test))
    random.Random(100).shuffle(c)
    y_test, y_pred, recon_err_test = zip(*c)
    viz.draw_anomaly(y_test, recon_err_test, threshold[i])



def kd_show(teacher_history, student_history, student_kd_history):
    x = list(range(1, local_epochs + 1))

    plt.plot(x, [teacher_history[i] for i in range(local_epochs)], label="teacher")
    plt.plot(x, [student_history[i] for i in range(local_epochs)], label="student")
    plt.plot(x, [student_kd_history[i] for i in range(local_epochs)], label="student_kd")

    plt.title("Test loss")
    plt.legend(loc='upper left')

    plt.show()

import itertools
from sklearn.metrics import confusion_matrix, recall_score
fontsize = 28
#### 混淆矩阵
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=fontsize)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label', fontsize=fontsize)
    plt.xlabel('Predicted label', fontsize=fontsize)
    plt.show()


if __name__ == '__main__':
    #summary(AutoEncoder_Teacher())
    teacher_model, teacher_history, teacher_acc = teacher_main()
    # student_model, student_history, student_acc  = student_main()
    # student_kd_model, student_kd_history, student_kd_acc= student_kd_main()
    # print("teacher_history :",teacher_history)
    # print("student_history :", student_history)
    # print("student_kd_history :", student_kd_history)
    #
    # print("teacher_acc :", teacher_acc)
    # print("student_acc :", student_acc)
    # print("student_kd_acc :", student_kd_acc)
    # kd_show(teacher_history, student_history, student_kd_history)
    # kd_show(teacher_acc, student_acc, student_kd_acc)
    #train_teacher( teacher_model, torch.optim.Adam(teacher_model.parameters(), 0.001),10)
    recon_err_test = err()
    threshhold(y_test, recon_err_test)