from os import listdir
import numpy as np
import operator


def classify0(inX, dataSets, labels, k):
    dataSeteSize = dataSets.shape[0]
    diffMat = np.tile(inX, (dataSeteSize, 1)) - dataSets
    sqDiffMAt = diffMat ** 2
    sqDistance = sqDiffMAt.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistIndicies = distance.argsort()
    classCount = {}

    for i in range(k):
        votelable = labels[sortedDistIndicies[i]]
        classCount[votelable] = classCount.get(votelable, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def image2vector(filename):
    res = np.zeros((1, 1024))
    file = open(filename)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            res[0, 32 * i + j] = int(line[j])
    return res


labels = []
trainFilelist = listdir("trainingDigits")
m = len(trainFilelist)
traininfMat = np.zeros((m, 1024))  # because of the  length of each num matrix is 1024
for i in range(m):
    filenamestr = trainFilelist[i]
    filestr   filenamestr.split(".")[0]
    numstr = filestr.split("_")[0]
    labels.append(int(numstr))
    traininfMat[i, :] = image2vector('trainingDigits/%s' % filenamestr)
testFilelist = listdir("testDigits")
error = 0
testnum = len(testFilelist)
for i in range(testnum):
    test_filenamestr = testFilelist[i]
    filestr = test_filenamestr.split(".")[0]
    numstr = filestr.split("_")[0]
    print(test_filenamestr)
    vector_test = image2vector("testDigits/%s" % test_filenamestr)
    res = classify0(vector_test, traininfMat, labels, 2)
    print("the class result is %d ,the real num is %d" % (res, int(numstr)))
    if res != int(numstr): error += 1
print("the error rate is %d" % (error /testnum))
