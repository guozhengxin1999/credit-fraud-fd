# import matplotlib.pyplot as plt
# import pandas as pd
# data = pd.read_csv('creditcard-discipline/creditcard.csv')
# # print(data.shape)
# # print(data.columns)
# # print(data.describe())
#
# data.hist(figsize=(30,30))
# plt.show()

G = (100 * a + 10 * b + c for a in range(0, 10)
     for b in range(0, 10)
     for c in range(0, 10)
     if a <= b <= c)
print(type(G))
print(sum(G))
print(sum(G))



print(G)