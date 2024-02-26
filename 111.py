
import pandas as pd
import tensorflow_privacy as tfp
import numpy as np

# 创建数据集
data = np.array([1, 2, 3, 4, 5])

# 初始化差分隐私查询
dp_query = tfp.NoPrivacySumQuery()

# 计算求和并添加噪声
epsilon = 0.5
noisy_sum = dp_query(data, epsilon=epsilon)

# 输出结果
print("原始数据：", data)
print("添加噪声后的求和结果：", noisy_sum)



# 读取人口统计数据集
df = pd.read_csv('population_data.csv')

# 将敏感列提取为numpy数组
sensitive_data = df['income'].to_numpy()

# 初始化差分隐私查询
dp_query = tfp.NoPrivacySumQuery()

# 定义隐私参数
epsilon = 1.0


# 定义查询函数
def query(data):
    return dp_query(data, epsilon=epsilon)

# 对敏感数据进行查询并添加噪声
noisy_result = query(sensitive_data)

# 输出结果
print("原始数据：")
print(sensitive_data)
print("添加噪声后的查询结果：")
print(noisy_result)