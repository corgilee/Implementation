###############################
# machine learning in action, p #209
###############################

import numpy as np
import sys
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
        p1 = minJ
        p2 = rangeJ * np.random.rand(k, 1)
        centroids[:, j] = p1+p2 # k rows and 1 col
    return centroids

def kMeans(dataSet, k ,distMeas=distEclud,createCent = randCent):
    m = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            # loop for each ob in the dataSet, row i
            minDist = sys.maxsize
            minIndex = -1
            for j in range(k):
                # loop for each centroid, centroids, every row (j), has a vector for the certain centroids
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            #check if ob i's group keep same
            if clusterAssment[i, 0] !=minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2  # update index, mindist
        for cent in range(k):
            # np.nonzero, check the notes in onenote
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # get the row/index number of of the none-zero row
        centroids[cent, :] = np.mean(ptsInClust, axis=0)  # average the vector along row direction
    return centroids, clusterAssment


if __name__ == "__main__":
    datMat = loadDataSet('testSet.txt')
    myCentroids, clustAssing = kMeans(datMat, 4)
    print(myCentroids)
    print(clustAssing)

