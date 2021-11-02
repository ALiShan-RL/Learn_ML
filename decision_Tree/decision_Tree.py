import operator
from math import log
import treePlotter
# 计算熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]   # 取最后一个标签 ，在myDat中就是取yes or no
        if currentLabel not in labelCounts.keys():  # 如果labelCounts没有这个标签，则创建
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0   # 计算累和
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算概率
        shannonEnt -= prob * log(prob,2)
    return shannonEnt


# 由于matplot中文乱码
# def createDataSet():
#     dataSet = [['青年', '否', '否', '一般', 'no'],
#                ['青年', '否', '否', '好', 'no'],
#                ['青年', '是', '否', '好', 'yes'],
#                ['青年', '是', '是', '一般', 'yes'],
#                ['青年', '否', '否', '一般', 'no'],
#                ['中年', '否', '否', '一般', 'no'],
#                ['中年', '否', '否', '好', 'no'],
#                ['中年', '是', '是', '好', 'yes'],
#                ['中年', '否', '是', '非常好', 'yes'],
#                ['中年', '否', '是', '非常好', 'yes'],
#                ['青年', '否', '是', '非常好', 'yes'],
#                ['青年', '否', '是', '好', 'yes'],
#                ['青年', '是', '否', '好', 'yes'],
#                ['青年', '是', '否', '非常好', 'yes'],
#                ['青年', '否', '否', '一般', 'no']]
#     labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
#     return dataSet,labels


# 生出数据集
def createDataSet():
    dataSet = [['young', 'no', 'no', 'common', 'no'],
               ['young', 'no', 'no', 'good', 'no'],
               ['young', 'yes', 'no', 'good', 'yes'],
               ['young', 'yes', 'yes', 'common', 'yes'],
               ['young', 'no', 'no', 'common', 'no'],
               ['middle', 'no', 'no', 'common', 'no'],
               ['middle', 'no', 'no', 'good', 'no'],
               ['middle', 'yes', 'yes', 'good', 'yes'],
               ['middle', 'no', 'yes', 'very', 'yes'],
               ['middle', 'no', 'yes', 'very', 'yes'],
               ['young', 'no', 'yes', 'very', 'yes'],
               ['young', 'no', 'yes', 'good', 'yes'],
               ['young', 'yes', 'no', 'good', 'yes'],
               ['young', 'yes', 'no', 'very', 'yes'],
               ['young', 'no', 'no', 'common', 'no']]
    labels = ['age', 'work', 'house', 'credit']
    return dataSet,labels

# 将数据集按照特征axis，并且该特征的值为value进行划分
def splitDataSet(dataSet, axis, value):     # 使用了三个参数，待划分的数据集、划分的数据集特征、需要返回的特征的值
    retDataSet = []  # 返回特征值为value的数据集
    for featVec in dataSet:
        if featVec[axis] == value:  # 如果特征值为value则
            reducedFeatVec = featVec[:axis]   # 将特征featVec[axis]刨除
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 运用信息增益选择最好的特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1     # 特征的数量 要减去最后的标签
    baseEntropy = calcShannonEnt(dataSet)  # 计算大数据集的熵
    bestInfoGain = 0.0;  bestFeature = -1
    for i in range(numFeatures):   # 第i个特征
        featList = [example[i] for example in dataSet]  # 取第i 个特征所有的值
        uniqueVals = set(featList)                      # 使用set容器将值进行排序和缩减
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 计算分割数据集的熵，熵描述的是事件的不确定性，当熵越大，事件的不确定性就越大
        infoGain = baseEntropy - newEntropy    # 求信息增益
        if(infoGain > bestInfoGain):           # 信息增益大的特征选为首要特征
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 创建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):    # 如果类型一致则停止划分
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)   # 选取最好的特征
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])           # 删除特征，因为剩下的集合是抛出了当前选择的特征的
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value), subLabels)  #递归创建树
    return myTree




myDat,labels = createDataSet()
print('myDat的数据为',(myDat))
print('myData的熵为' ,(calcShannonEnt(myDat)))

print(chooseBestFeatureToSplit(myDat))  # 输出最好的特征


myTree = createTree(myDat, labels)
print(myTree)
treePlotter.createPlot(myTree)
