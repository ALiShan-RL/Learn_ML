from math import log
def calcShannonEnt(dataSet):  # 计算熵
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

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels


def splitDataSet(dataSet, axis, value):     # 使用了三个参数，待划分的数据集、划分的数据集特征、需要返回的特征的值
    retDataSet = []  # 返回特征值为value的数据集
    for featVec in dataSet:
        if featVec[axis] == value:  # 如果特征值为value则
            reducedFeatVec = featVec[:axis]   # 将特征featVec[axis]刨除
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

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




myDat,labels = createDataSet()
print('myDat的数据为',(myDat))
print('myData的熵为' ,(calcShannonEnt(myDat)))

print(splitDataSet(myDat,0,1))
print(splitDataSet(myDat,0,0))
print(chooseBestFeatureToSplit(myDat))  # 输出最好的特征