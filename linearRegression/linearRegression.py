# LinearRegression线性回归
from sklearn import datasets, linear_model

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.model_selection import train_test_split


houseDataSets = datasets.load_boston() # 利用sklearn中的房价数据


# # 使用了所有的参数，多元线性规划
# data_X = houseDataSets.data     # np.newaxis是将数据增加维数，选择第13列的数据
# data_Y = houseDataSets.target   # 输出数据
# train_X, test_X,  train_Y, test_Y = train_test_split(data_X, data_Y, test_size=0.3)   # 70%的数据用来训练模型  30%数据用来测试模型
#
# # * fit_intercept : 布尔型参数，表示是否计算该模型截距。可选参数。
# # * normalize : 布尔型参数，若为True，则X在回归前进行归一化。可选参数。默认值为False。
# # * copy_X : 布尔型参数，若为True，则X将被复制；否则将被覆盖。 可选参数。默认值为True。
# # * n_jobs : 整型参数，表示用于计算的作业数量；若为-1，则用所有的CPU。可选参数。默认值为1。
# regr = linear_model.LinearRegression(normalize=True)
#
# # 使用的是最小化残差平方和（residual sum of squares）
# regr.fit(train_X, train_Y)        # 线性回归训练
#
# a = regr.intercept_  # 截距
# b = regr.coef_  # 回归系数
# print("拟合参数:截距", a, ",回归系数：", b)
# predict_Y = regr.predict(test_X)  # 将测试集的数据带入模型中，求出预测数据
#
# # 分数为1代表最好
# sorce = regr.score(test_X, test_Y)
# print(sorce)
#
# plt.xlabel('range(len(predict_Y))', fontsize=12)
# plt.ylabel('house price', fontsize=12)
# plt.plot(range(len(predict_Y)), predict_Y, 'red', linewidth=2, label="predict data")
# plt.plot(range(len(test_Y)), test_Y, 'green', label="test data")
#
# plt.show()


# houseDataSets = datasets.load_boston() # 利用sklearn中的房价数据
#
# print(houseDataSets)
# # 只是用了第13列的特征，一元线性规划
#data_X = houseDataSets.data[:, np.newaxis,12]     # np.newaxis是将数据增加维数，选择第13列的数据
data_X = houseDataSets.data[:, np.newaxis,12]
data_Y = houseDataSets.target  # 输出数据
train_X, test_X,  train_Y, test_Y = train_test_split(data_X, data_Y, test_size=0.3)   # 70%的数据用来训练模型  30%数据用来测试模型

# epoch = 100
# alpha = 0.1
# theta = np.matrix([0]) # 初始化参数向量
#
# temp = np.zeros(theta.shape)
# X = np.matrix(train_X)
# Y = np.matrix(train_Y)
# X1 = np.matrix(test_X)
# cost = np.zeros(epoch)
# m = train_X.shape[0]

# for i in range(epoch):
#     print(1111)
#     print(theta)
#     #print(Y.shape)
#     theta = theta - (alpha/X.shape[0]) * (X * theta.T - Y).T * X
    #theta = temp

   # cost[i] = np.power(((train_X * theta.T) - train_Y), 2)
   #

regr = linear_model.LinearRegression()
regr.fit(train_X, train_Y)        # 线性回归训练

predict_Y = regr.predict(test_X)  # 将测试集的数据带入模型中，求出预测数据
print(predict_Y.shape)
# 画图
from itertools import chain
# predict_Y = X1 * theta
#print(predict_Y)
#print(predict_Y1)
mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.scatter(test_X, test_Y, color='black')    # 将测试集用点显示出来
plt.plot(test_X, predict_Y, color='red', linewidth=1) # 画出预测的先

plt.xlabel('lower status of the population', fontsize=12)
plt.ylabel('house price', fontsize=12)
plt.title(u'LinearRegression', fontsize=12)
plt.show()
#












# diabetes = datasets.load_diabetes()
# print(diabetes)
# diabetes_x = diabetes.data[:, np.newaxis, 2]  # 取第三列数据
# print(diabetes_x)
#
# diabetes_x_train = diabetes_x[:-20]
# diabetes_x_test = diabetes_x[-20:]
#
#
#
# diabetes_y_train = diabetes.target[:-20]
# diabetes_y_test = diabetes.target[-20:]
#
# # 核心代码
# regr = linear_model.LinearRegression()
# regr.fit(diabetes_x_train, diabetes_y_train)  # 用训练集进行训练模型
#
# print('Input Values')
# print(diabetes_x_test)
#
# # 核心代码
# diabetes_y_pred = regr.predict(diabetes_x_test)
# print('Predicted Output Values')
# print(diabetes_y_pred)
#
# # 绘图
# mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 用来正常显示中文标签
# mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
# plt.scatter(diabetes_x_test, diabetes_y_test, color='black')
# plt.plot(diabetes_x_test, diabetes_y_pred, color='red', linewidth=1)
#
# plt.xlabel('体质指数', fontsize=12)
# plt.ylabel('一年后患疾病的定量指标', fontsize=12)
# plt.title(u'LinearRegression线性回归', fontsize=12)
#
#