#########################################
# machine learning in action, p 67
#########################################
import numpy as np


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


###### step1: volcabulory #######
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

def setOfWords2Vec (vocabList, inputSet):
    '''
    take the vocabulary list and a document and outputs a vector of 1/0 s
    :param vocabList:
    :param inputSet:
    :return:
    '''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word {} is not in my Vocabulary...".format(word))
    return returnVec

##############################################
# Naive Bayes classifier training function
##############################################

def trainNB0(trainMatrix, trainCategory):
    # number of training obs and Words
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    # trainCategory is the vector of category
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    # Initialize probabilities, p(w/c)
    p0Num, p1Num =np.ones(numWords), np.ones(numWords)
    # p0Denom is the sum of count of all the words in class 0
    p0Denom, p1Denom =2.0, 2.0

    # Vector addition
    for i in range(numTrainDocs):
        if trainCategory[i] ==1:
            p1Num += trainMatrix[i] # add the trainMatrix vector
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i] # add the trainMatrix vector
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom) # np.log for each element in the vector
    p0Vect = np.log(p0Num/p0Denom)

    return p0Vect, p1Vect, pAbusive

# use log(x) instead of x to avoid "underflow"
def classifyNB(vec2Classify,p0Vec,p1Vec, pClass1):
    '''

    :param vec2Classify: array of the wordvec
    :param p0Vec: array of pword in class0
    :param p1Vec: array of pword in class0
    :param pClass1:
    :return:
    '''

    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)


    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses= loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat =[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))

    print('{}, classified as: {}'.format(testEntry, classifyNB(thisDoc, p0V, p1V, pAb)))

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print('{}, classified as: {}'.format(testEntry, classifyNB(thisDoc, p0V, p1V, pAb)))


if __name__ =="__main__":

    testingNB()





