import math

import pandas as pd
import matplotlib.pyplot as plt
from numpy.ma import copy

plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np
# 读入数据
data = pd.read_csv("creditcard.csv")
data.head()
pd.value_counts(data['Class'], sort=True)

from sklearn.preprocessing import StandardScaler  # 标准化模块
# 标准化
data['normAmount'] = StandardScaler().fit_transform(data.Amount.values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)  # 删除不需要的列
data.head()



######################## 下采样 ########################
X = data.iloc[:, data.columns != 'Class']  # 特征数据
y = data.iloc[:, data.columns == 'Class']  # 标签数据

number_records_fraud = len(data[data.Class == 1])  # 异常样本数量
fraud_indices = data[data.Class == 1].index  # 得到所有异常样本的索引
normal_indices = data[data.Class == 0].index  # 得到所有正常样本的索引

# 在正常样本中随机采样
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)

# 根据索引得到下采样所有样本
under_sample_data = data.iloc[np.concatenate([fraud_indices, random_normal_indices]), :]
X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']  # 特征数据
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']  # 标签数据

pd.value_counts(under_sample_data['Class'], sort=True)  # 观察数据



## 测试集 训练集
from sklearn.model_selection import train_test_split
# 对原始数据集进行划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
len(X_train)  # 原始训练集包含样本数量
len(X_test)  # 原始测试集包含样本数量

print("x_train", len(X_train))

# 对下采样数据集进行划分
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,
                                                                                                    y_undersample,
                                                                                                    test_size=0.3,
                                                                                                    random_state=0)
len(X_train_undersample)  # 下采样训练集包含样本数量
len(X_test_undersample)  # 下采样测试集包含样本数量




from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, recall_score

## K折交叉验证
def printing_Kfold_scores(x_train_data, y_train_data):
    # k-fold表示K折的交叉验证，会得到两个索引集合: 训练集 = indices[0], 验证集 = indices[1]
    fold = KFold(5, shuffle=False)

    recall_accs = []
    for iteration, indices in enumerate(fold.split(x_train_data)):
        # 实例化算法模型，指定l1正则化
        lr = LogisticRegression(C=0.01, penalty='l1', solver='liblinear')

        # 训练模型，传入的是训练集，所以X和Y的索引都是0
        lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

        # 建模后，预测模型结果，这里用的是验证集，索引为1
        y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

        # 评估召回率，需要传入真实值和预测值
        recall_acc = round(recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample), 4)
        recall_accs.append(recall_acc)
        print('第', iteration + 1, '次迭代：召回率 = ', recall_acc)

    # 当执行完交叉验证后，计算平均结果
    print('平均召回率 ', round(np.mean(recall_accs), 4))

    return None



import itertools
fontsize=30
#### 混淆矩阵
def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize = fontsize)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小。
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0,fontsize = 20)
    plt.yticks(tick_marks, classes,fontsize = 20)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize = 20)
    plt.tight_layout()
    plt.ylabel('True label',fontsize=fontsize)
    plt.xlabel('Predicted label',fontsize=fontsize)
    plt.show()


# 下采样训练集训练之后，预测原始测试集
def draw_train_undersample():
    lr = LogisticRegression(C=0.01, penalty='l1', solver='liblinear')
    lr.fit(X_train_undersample, y_train_undersample.values.ravel())
    y_pred = lr.predict(X_test.values)
    print("召回率: ", recall_score(y_test.values, y_pred))
    # 绘制混淆矩阵
    plot_confusion_matrix(confusion_matrix(y_test.values, y_pred), [0, 1], "SMOTE")


# 下采样训练集训练之后，预测原始测试集
def draw_sample():
    lr = LogisticRegression(C=0.01, penalty='l1', solver='liblinear')
    lr.fit(X_train, y_train.values.ravel())
    y_pred = lr.predict(X_test.values)
    print("召回率: ", recall_score(y_test.values, y_pred))
    # 绘制混淆矩阵
    plot_confusion_matrix(confusion_matrix(y_test.values, y_pred), [0, 1], "Raw Data")

#smote
from imblearn.over_sampling import SMOTE

def draw_smote():
    oversampler = SMOTE(random_state=0)
    os_data, os_labels = oversampler.fit_resample(X_train, y_train)
    pd.value_counts(os_labels.Class)
    lr = LogisticRegression(C=0.01, penalty='l1', solver='liblinear')
    lr.fit(os_data, os_labels.values.ravel())
    os_pred = lr.predict(X_test.values)
    print("召回率: ", recall_score(y_test.values, os_pred))
    plot_confusion_matrix(confusion_matrix(y_test, os_pred), [0, 1], "Custom SMOTE")


# lon和lat分别是要插值的点的x,y
# lst是已有数据的数组，结构为：[[x1，y1，z1]，[x2，y2，z2]，...]
# 返回值是插值点的高程
def interpolation(lon, lat, lst):
    p0 = [lon, lat]
    sum0 = 0
    sum1 = 0
    temp = []
    P = 2
    # 遍历获取该点距离所有采样点的距离
    for point in lst:
        if lon == point[0] and lat == point[1]:
            return point[2]
        Di = distance(p0, point)
        # new出来一个对象，不然会改变原来lst的值
        ptn = copy.deepcopy(point)
        ptn.append(Di)
        temp.append(ptn)

    # 根据上面ptn.append（）的值由小到大排序
    temp1 = sorted(temp, key=lambda point: point[3])
    # 遍历排序的前15个点，根据公式求出sum0 and sum1
    for point in temp1[0:15]:
        sum0 += point[2] / math.pow(point[3], P)
        sum1 += 1 / math.pow(point[3], P)
    return sum0 / sum1


# 计算两点间的距离
def distance(p, pi):
    dis = (p[0] - pi[0]) * (p[0] - pi[0]) + (p[1] - pi[1]) * (p[1] - pi[1])
    m_result = math.sqrt(dis)
    return m_result


if __name__ == '__main__':
     #draw_train_undersample()
   #  draw_sample()
     #draw_smote()
     printing_Kfold_scores(X_train, y_train)
     printing_Kfold_scores(X_train_undersample, y_train_undersample)
