import math

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
import torch.utils.data
import torchvision
from torchinfo import summary # 可视化模型结构
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

epochs = 5
batch_size = 64

#设置随机种子，方便复现
torch.manual_seed(0)
torch.cuda.manual_seed(0)


# 训练教师网络
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(p=0.3)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x) # 没经过其他操作（softmax）
        return output


# 定义训练函数
def train_teacher(model, device, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        trained_samples += len(data)
        progress = math.ceil(batch_idx/len(train_loader)*50)
        print("\rTrain epoch %d:%d%d,[%-51s]%d%%" %
          (epoch, trained_samples, len(train_loader.dataset),
           "-" * progress + '>', progress * 2), end='')


# 定义测试函数
def test_teacher(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()  # 总结批次损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大对数概率索引
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("\nTest:average loss:{:.4f},accuracy:{}/{}({:.0f}%)".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)


# 训练教师网络
def teacher_main():
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    train_loader=torch.utils.data.DataLoader(
        datasets.MNIST("../data/MNIST", train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data/MNIST", train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081, ))
                        ])),
        batch_size=1000, shuffle=True)

    model = TeacherNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())

    teacher_history=[]

    for epoch in range(1, epochs+1):
        train_teacher(model, device, train_loader, optimizer, epoch)
        loss, acc = test_teacher(model, device, test_loader)

        teacher_history.append((loss, acc))
    torch.save(model.state_dict(), "teacher.pt")
    #show(teacher_history, "teacher")
    return model, teacher_history


def softmax_t(x, t):
    x_exp = np.exp(x / t)
    return x_exp / np.sum(x_exp)


def show(history, title):
    x=list(range(1, epochs+1))

    #测试精度
    plt.subplot(2, 1, 1)
    plt.plot(x, [history[i][1] for i in range(epochs)], label=title)

    plt.title("Test accuracy")
    plt.legend()

    #测试损失
    plt.subplot(2, 1, 2)
    plt.plot(x, [history[i][0] for i in range(epochs)], label=title)

    plt.title("Test loss")
    plt.legend()

    plt.show()


#####################################################
# 学生网络
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1=nn.Linear(28 * 28, 128)
        self.fc2=nn.Linear(128, 64)
        self.fc3=nn.Linear(64, 10)

    def forward(self, x):
        x=torch.flatten(x, 1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        output=F.relu(self.fc3(x))
        return output

# 学生训练网络
def train_student(model, device, train_loader, optimizer, epoch):
    model.train()
    trained_samples=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target=data.to(device), target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        trained_samples += len(data)
        progress=math.ceil(batch_idx/len(train_loader)*50)
        print("\rTrain epoch %d:%d%d,[%-51s]%d%%" %
              (epoch, trained_samples, len(train_loader.dataset),
               "-" * progress + '>', progress * 2), end='')


def test_student(model, device, test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target=data.to(device), target.to(device)
            output=model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item() #总结批次损失
            pred=output.argmax(dim=1, keepdim=True) # 获取最大对数概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("\nTest:average loss:{:.4f},accuracy:{}/{}({:.0f}%)".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)


def student_main():
    torch.manual_seed(0)

    device=torch.device("cuba" if torch.cuda.is_available() else "cpu")

    #加载数据集
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data/MNIST", train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data/MNIST", train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=True)

    model = StudentNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())

    student_history = []
    for epoch in range(1, epochs + 1):
        train_student(model, device, train_loader, optimizer, epoch)
        loss, acc = test_student(model, device, test_loader)
        student_history.append((loss, acc))
    torch.save(model.state_dict(), "student.pt")
    #show(student_history,"student")
    return model, student_history


##########################################################################################################
#蒸馏模型的损失函数
def distillation(y, labels, teacher_scores, temp, alpha):
    return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


# 定义蒸馏模型的训练，测试函数
def train_student_kd(model, device, train_loader, optimizer, epoch):
    model.train()
    trained_samples=0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():  # 教师网络不用反向传播
            teacher_output = teacher_model(data)

        # 学生网络训练
        student_output = model(data)
        loss=distillation(student_output, target, teacher_output, temp=5.0, alpha=0.7)
        loss.backward()
        optimizer.step()

        trained_samples += len(data)
        progress=math.ceil(batch_idx/len(train_loader)*50)
        print("\rTrain epoch %d:%d%d,[%-51s]%d%%" %
              (epoch, trained_samples, len(train_loader.dataset),
               "-" * progress + '>', progress * 2), end='')


def test_student_kd(model, device, test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target=data.to(device), target.to(device)
            output=model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred=output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("\nTest:average loss:{:.4f},accuracy:{}/{}({:.0f}%)".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)

#训练蒸馏函数
def student_kd_main():
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data/MNIST", train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data/MNIST", train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=True)

    model = StudentNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())

    student_kd_history = []

    for epoch in range(1, epochs + 1):
        train_student_kd(model, device, train_loader, optimizer, epoch)
        loss, acc = test_student_kd(model, device, test_loader)
        student_kd_history.append((loss, acc))
    torch.save(model.state_dict(), "student_kd.pt")
    #show(student_kd_history, "student_kd")
    return model, student_kd_history


def kd_show(teacher_history, student_history, student_kd_history):
    x = list(range(1, epochs + 1))
    # 测试的精度
    plt.subplot(2, 1, 1)
    plt.plot(x, [teacher_history[i][1] for i in range(epochs)], label="teacher")
    plt.plot(x, [student_history[i][1] for i in range(epochs)], label="student")
    plt.plot(x, [student_kd_history[i][1] for i in range(epochs)], label="student_kd")

    plt.title("Test accuracy")
    plt.legend()

    # 测试的损失
    plt.subplot(2, 1, 2)
    plt.plot(x, [teacher_history[i][0] for i in range(epochs)], label="teacher")
    plt.plot(x, [student_history[i][0] for i in range(epochs)], label="student")
    plt.plot(x, [student_kd_history[i][0] for i in range(epochs)], label="student_kd")

    plt.title("Test loss")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    teacher_model, teacher_history = teacher_main()
    student_model, student_history = student_main()
    student_kd_model, student_kd_history = student_kd_main()
    kd_show(teacher_history, student_history, student_kd_history)