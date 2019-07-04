#########################################
# machine learning in action, p 67
#########################################

###### Word list to vector function #######
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', \
                    'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', \
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', \
                    'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', \
                    'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec=[0, 1, 0, 1, 0, 1] # 1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet):
    '''
    Create a list of all unique words in all of our documents
    :param dataSet:
    :return:
    '''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) # use '|' to create the union of two sets
    return list(vocabSet)

############ Words2Vec ############

def setofWords2Vec (vocabList, inputSet):
    '''
    take the vocabulary list and a document and outputs a vector of 1/0 s
    :param vocabList:
    :param inputSet:
    :return:
    '''
    returnVec= [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word {} is not in my Vocabulary...".format(word))
    return returnVec


