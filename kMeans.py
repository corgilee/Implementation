###############################
# machine learning in action, p #209
###############################

import numpy as np
import math


# support functions, read files

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(list(fltLine))
    return np.mat(dataMat)


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    '''
    creates a set of k random (initial) centroids need to be within the bounds of the dataset
    The random centroids need to be within the bounds of the dataset.
    :param dataSet:
    :param k:
    :return: initialize centroid (k,n)
    '''
    n = dataSet.shape[1]  # number of features
    centroids = np.mat(np.zeros((k, n)))  # k rows, n features
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)  # k rows and 1 col
    return centroids


if __name__ == "__main__":
    dataMat = loadDataSet('testSet.txt')
    print(randCent(dataMat, 2))
    print(distEclud(dataMat[0], dataMat[1]))
