# 本项目是根据《机器学习实战》上面的朴素贝叶斯进行练习

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


listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
print(myVocabList)
print(setOfWords2Vec(myVocabList, listOPosts[0]))