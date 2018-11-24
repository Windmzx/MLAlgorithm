# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np

from KNN.main import classify0
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def file2matrix(filename):
    fr=open(filename,"r")
    lines=fr.readlines()
    linenum=len(lines)
    returnmat=np.zeros((linenum,3))
    classLableVector=[]
    index=0
    for line in lines:
        line=line.strip().split('\t')
        returnmat[index,:]=line[0:3]
        classLableVector.append(int(line[-1]))
        index+=1
    return returnmat,classLableVector

def draw(data,lables):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(data[:,0],data[:,1],15.0*np.array(lables),15.0*np.array(lables))
    plt.xlabel(u"每年获取的飞行常客里程数")
    plt.ylabel(u"玩视频游戏所占百分比")
    plt.savefig("data12.jpg")
    plt.show()
def autoNorm(dataSet):
    minvals=dataSet.min(0)
    maxvals=dataSet.max(0)
    ranges=maxvals-minvals
    normDataset=np.zeros(np.shape(dataSet))
    m=dataSet.shape[0]
    normDataset=dataSet-np.tile(minvals,(m,1))
    normDataset=normDataset/np.tile(ranges,(m,1))
    return normDataset,ranges,minvals

def classTst():
    hoRation=0.1
    datingDataMat,datingLable=file2matrix("datingTestSet2.txt")
    normMat,ranges,minvals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numtest=int(m*hoRation)
    errorCount=0.0
    for i in range(numtest):
        classifierResult=classify0(normMat[i,:],normMat[numtest:m,:],datingLable[numtest:m],3)
        print("the classifier came back with %d,the real answer is :%d"%(classifierResult,datingLable[i]))
        if(classifierResult!=datingLable[i]):
            errorCount+=1
    print("the total error rate is :%f"%(errorCount/float(numtest)))



if __name__ == '__main__':
    classTst()