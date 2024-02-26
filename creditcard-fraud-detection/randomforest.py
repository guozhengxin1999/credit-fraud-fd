import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt



epochs = 10

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


# # 用默认参数训练
# rf0 = RandomForestClassifier(oob_score=True, random_state=666)
# rf0.fit(X_train, y_train.values.ravel())
# print(rf0.oob_score_)
# y_predprob = rf0.predict_proba(X_test)[:, 1]
# print("AUC Score (Train): %f" % roc_auc_score(y_test, y_predprob))

# 随机树由的复杂程度和训练准确程度是由以下三个参数决定：
# max_feature（生成单个决策树时的特征数，提高单个决策模型的性能）
# n_estimators（决策树的棵树，较多的子树让模型拥有更好的稳定性和泛化能力）
# max_depth（树深，树深太大可能会过拟合，模型样本量多特征多，会限制最大树深）

# 构建教师网络
def teacher_net():
    print("########################### Teacher Net #################################")
    # 优化n_estimators
    print("############################ 优化n_estimators ############################")
    param_test1 = {'n_estimators': range(10, 101, 10)}
    gsearch1 = GridSearchCV(estimator=RandomForestClassifier(random_state=666, n_jobs=2),
                            # oob_score=True 去掉了这个参数，warning
                            param_grid=param_test1, scoring='roc_auc', cv=5)
    gsearch1.fit(X_train, y_train.values.ravel())
    #print(gsearch1.cv_results_)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)

    # 优化max_depth
    print("############################ 优化max_depth ############################")
    param_test2 = {'max_depth': range(2, 12, 2)}
    gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=gsearch1.best_params_.get('n_estimators'),
                                                             oob_score=True, random_state=666, n_jobs=2),
                            param_grid=param_test2, scoring='roc_auc', cv=5)
    gsearch2.fit(X_train, y_train.values.ravel())
   # print(gsearch2.cv_results_)
    print(gsearch2.best_params_)
    print(gsearch2.best_score_)

    # 优化max_feature
    print("############################ 优化max_feature ############################")
    param_test3 = {'min_samples_split': range(2, 8, 1)}
    gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=gsearch1.best_params_.get('n_estimators'),
                                                             max_depth=gsearch2.best_params_.get('max_depth'),
                                                             oob_score=True, random_state=666, n_jobs=2),
                            param_grid=param_test3, scoring='roc_auc', cv=5)
    gsearch3.fit(X_train, y_train.values.ravel())
   # print(gsearch3.cv_results_)
    print(gsearch3.best_params_)
    print(gsearch3.best_score_)

    # 教师模型
    print("############################综合测试############################")
    teacher = RandomForestClassifier(n_estimators=gsearch1.best_params_.get('n_estimators'),
                                     max_depth=gsearch2.best_params_.get('max_depth'),
                                     min_samples_split=gsearch3.best_params_.get('min_samples_split'),
                                     oob_score=True, random_state=666, n_jobs=2)
    return teacher


# 训练教师网络
def teacher_main():

    teacherrf = teacher_net()
    teacherrf.fit(X_train, y_train.values.ravel())
    y_predprob = teacherrf.predict_proba(X_test)[:, 1]  # 概率
    print("AUC Score (Train): %f" % roc_auc_score(y_test, y_predprob))
    return teacherrf
    # teacher_history = []
    # for epoch in range(1, epochs+1):
    #     teacherrf.fit(X_train, y_train.values.ravel())
    #     #print(teacherrf.oob_score_)
    #     y_predprob = teacherrf.predict_proba(X_test)[:, 1]  # 概率
    #     y_pred = teacherrf.predict(X_test)
    #     y_test_change = np.squeeze(np.array(y_test.sort_index()))
    #     teacher_history.append(roc_auc_score(y_test, y_pred))
    #
    #     #standard(y_pred, y_test_change)
    # print(teacher_history)
    # x = list(range(1,epochs+1))
    # plt.plot(x, [teacher_history[i] for i in range(10)], label="Teacher")
    #
    # plt.legend()
    # plt.show()
    # print("AUC Score (Train): %f" % roc_auc_score(y_test, y_pred))


# 构建学生网络
def student_net():
    # student = RandomForestClassifier(n_estimators=n_estimators, max_depth=2, min_samples_split=2,
    #                                  oob_score=True, random_state=666)
    # return student
    rf0 = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_split=2, random_state=666)
    return rf0


# 训练学生网络
def student_main():
    print("########################### Student Net #################################")
    rf0 = student_net()
    rf0.fit(X_train, y_train.values.ravel())
    y_predprob = rf0.predict_proba(X_test)[:, 1]
    print("AUC Score (Train): %f" % roc_auc_score(y_test, y_predprob))

    # student_history=[]

    # for epoch in range(1, epochs+1):
    #     student = student_net(epoch+10)
    #     student.fit(X_train, y_train.values.ravel())
    #     y_predprob = student.predict_proba(X_test)[:, 1]  # 概率
    #    # y_pred = student.predict(X_test)
    #    # y_test_change = np.squeeze(np.array(y_test.sort_index()))
    #     student_history.append(roc_auc_score(y_test, y_predprob))
    #
    # print(student_history)
    # x = list(range(1, epochs + 1))
    # plt.plot(x, [student_history[i] for i in range(10)], label="Student")
    #
    # plt.legend()
    # plt.show()
    # standard(y_pred, y_test_change)
   # print("AUC Score (Train): %f" % roc_auc_score(y_test, y_predprob))


def softmax_t(x, t):
    x_exp = np.exp(x / t)
    return x_exp / np.sum(x_exp)

# 蒸馏模型的损失函数,这块需要重新定义
def distillation(y, labels, teacher_scores, temp, alpha):
    return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), softmax_t(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


def train_student_kd():
    loss = distillation(student_output, y_test, teacher_output, temp=5.0, alpha=0.7)
    print(loss)


# 蒸馏模型
def student_kd_main():
    teacher_model / student_model


def standard(y_pred , y_test):
    true = np.sum(y_pred == y_test)
    print('预测对的结果数目为：', true)
    print('预测错的的结果数目为：', y_test.shape[0] - true)
    # 评估指标
    print('预测数据的准确率为： {:.4}%'.format(accuracy_score(y_test, y_pred) * 100))
    print('预测数据的精确率为：{:.4}%'.format(
        precision_score(y_test, y_pred) * 100))
    print('预测数据的召回率为：{:.4}%'.format(
        recall_score(y_test, y_pred) * 100))
    # print("训练数据的F1值为：", f1score_train)
    print('预测数据的F1值为：',
          f1_score(y_test, y_pred))
    print('预测数据的Cohen’s Kappa系数为：',
          cohen_kappa_score(y_test, y_pred))


if __name__ == '__main__':
    teacher_model = teacher_main()
    student_model = student_main()
    student_kd_main()
