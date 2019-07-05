import numpy as np
import operator

##############################################
## machine learning in action, p 23

##############################################
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels=['A','B','B','B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    '''
    :param inX: input vector
    :param dataSet: original dataset
    :param labels: vector of labels
    :param k: k
    :return:
    '''
    dataSetSize=dataSet.shape[0]
    #repeat in (row direction, col directions=1)
    diffMat=np.tile(inX, (dataSetSize, 1)) - dataSet #diffrence  of the matrix
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) #axis=1
    distance=sqDistances**0.5
    sortedDistIndicies=distance.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]] # get the labels from sorted index
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    # print(classCount)
    # print(sortedClassCount[0][0])
    return sortedClassCount[0][0]

###### text record to numpy parsing code #########
def file2matrix(filename):
    fr = open(filename) # 'open' build-in function
    numberOfLines = len(fr.readlines())
    returnMat=np.zeros((numberOfLines,3)) #create numpy matrix to return
    classLabelVector=[]

    fr=open(filename)
    index=0
    for line in fr.readlines():
        line = line.strip() # removing leading and trailing space in the string
        listFromLine=line.split('\t') #to list
        returnMat[index,:] = listFromLine[0:3] #there are 4 elements in each line
        classLabelVector.append(listFromLine[-1])
        index+=1

    return returnMat,classLabelVector #return matrix and label list

###### Data normalization code #########

def autoNorm(dataSet):
        minVals=dataSet.min(0) #along the row directions
        maxVals=dataSet.max(0)
        ranges=maxVals-minVals
        normDataSet = np.zeros(dataSet.shape) #create a zero matrix with the shape of dataset
        m=dataSet.shape[0] # number of rows
        normDataSet = dataSet-np.tile(minVals,(m,1)) #scale the minVals along the axis directions
        normDataSet = normDataSet/np.tile(ranges,(m,1))
        return normDataSet, ranges, minVals

def datingClassTest():
    '''
    use the first hoRatio of the datingDataMat as numTestVecs
    :return:
    '''
    hoRatio=0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                   datingLabels[numTestVecs:m],3)
        print("the classifier came back with: {}, the real answer is: {}" \
        .format(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print('the total error rate is: {}'.format(errorCount / float(numTestVecs)))



if __name__ == "__main__":
    datingClassTest()





# if __name__ == "__main__":
#     group, labels = createDataSet()
#     print('The group is classified as ',classify0([0,0], group, labels, 3))




