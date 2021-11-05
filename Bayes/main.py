# 本项目是根据《机器学习实战》上面的朴素贝叶斯进行练习
from numpy import *
def loadDataSet():
    postingList =[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                  ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
                  ]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常文字
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])                            # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)       # 创建两个集合的并集
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word:', (word), 'is not in my Vocabulary!')
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                           # 一共有6个词汇表，所以len(trainMatrix) = 6
    numWords = len(trainMatrix[0])                            # 第一组的单词个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)       # pAbusive = 侮辱性话个数 / 所有话的个数
    p0Num = ones(numWords)                                   # p0num 是一组值为1且个数与要检验的话相等
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                             # 如果第i个词汇表的词为侮辱性的词
            p1Num += trainMatrix[i]                           # 则p1Num 的向量就加上trainMatrix[i]
            p1Denom += sum(trainMatrix[i])                    # p1Denom 为 1，3，5 词汇表   与  trainMatrix 比较的侮辱性词汇的总和
        else:
            p0Num += trainMatrix[i]                           #
            p0Denom += sum(trainMatrix[i])                    # p0Denom = 24
    p1Vect = log(p1Num/p1Denom)                               # 防止因子太小，造成下溢
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)

    if p1 > p0:
        return 1
    else:
        return 0
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)        # 根据网上侮辱性的句子，做出自己的判断词性的词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(thisDoc)
    print('testEntry classified as: %d' %(classifyNB(thisDoc, p0V, p1V, pAb)))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print('testEntry classified as: %d' %(classifyNB(thisDoc,p0V,p1V,pAb)))

listOPosts, listClasses = loadDataSet()      # listOPosts代表词汇表，listClasses代表词汇表的分类
myVocabList = createVocabList(listOPosts)    # myVocabList代表自己创造的一些词

trainMat = []                                                 # 对词性的分类的结果保存,一共有6组分别是对6个词汇表进行比对的结果
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))    # 进行词的分类

p0V, p1V, pAb = trainNB0(trainMat, listClasses)
testingNB()