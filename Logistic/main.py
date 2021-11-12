from numpy import *
import matplotlib.pyplot as plt
import numpy as np
# 数据集一共有三列，第一列代表x，第二列代表y，第三列是这个点所属的类别

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('dataSet/testSet.txt')
    for line in fr.readlines():                        # 读取文件
        lineArr = line.strip().split()                 # 去除空格和分隔
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 1是为了bias添加了，其他两个元素为x和y
        labelMat.append(int(lineArr[2]))                             # 添加标签
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))                           # 制作sigmoid函数


def gradAscent(dataMatIn, classLabels):                # 梯度上升函数
    dataMatrix = mat(dataMatIn)                        # 将列表变成矩阵
    labelMat = mat(classLabels).transpose()            # transpose()默认是矩阵转置，与x.T一样的功能
    m, n = shape(dataMatrix)                           # m和n代表dataMatrix的行和列 m = 100 n = 3
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))                              # 初始化n行1列的权重为1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)                         # 计算误差
        weights = weights + alpha * dataMatrix.transpose() * error   # 梯度上升
    return weights

def plotBestFit(weights1, weights2, weights3):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='p')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y1 = (-weights1[0] - weights1[1] * x)/weights1[2]         # 1处设置了sigmoid 函数为0 ， 0 是两个分类的分解处
    y2 = (-weights2[0] - weights2[1] * x)/weights2[2]         # 因为当 w0x0 + w1x1 + w2x2 = 0 时，则sigmoid = 0.5正好为 分界线，因为x0=1 ，求解x1与x2的关系即可,因为刚开始
    y3 = (-weights3[0] - weights3[1] * x)/weights3[2]
    ax.plot(x, y1.T, ls='-', lw=2, label='batchAscent',color='green')
    ax.plot(x, y2.T, ls='-', lw=2, label='orderAscent', color='red')
    ax.plot(x, y3.T, ls='-', lw=2, label='stocAscent', color='blue')
    plt.legend()
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + np.array(dataMatrix[i]).astype(dtype=float) * alpha * error
    return weights


def stocGradAscent1(dataMatrix , classLabels, numIter=150):      # 改进的梯度上升算法
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):   # 进行150次梯度上升，而batch进行了500次，且batch每次用所有的元素进行参数更新，计算量较大
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01    # 动态更新alpha，使学习更稳定
            randomIndex = int(random.uniform(0, m))
            h = sigmoid(sum(dataMatrix[randomIndex] * weights))
            error = classLabels[randomIndex] - h
            #print(dataMatrix[i][0])
            weights = weights + np.array(dataMatrix[randomIndex]).astype(dtype=float) * alpha * error
            del(randomIndex)
    return weights



dataArr, labelMat = loadDataSet()
weights1 = gradAscent(dataArr, labelMat)
weights2 = stocGradAscent0(dataArr, labelMat)
weights3 = stocGradAscent1(dataArr, labelMat)
# print(weights)
plotBestFit(weights1, weights2, weights3)