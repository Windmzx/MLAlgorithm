import numpy as np
import operator

def creatTestData():
    groups=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lables=["A","A","B","B"]
    return groups,lables
def draw(data):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("data")
    plt.plot(data[:,0], data[:,1], 'ro')
    plt.savefig("data.jpg")
    plt.show()


def classify0(inX, dataSets, labels, k):
    dataSeteSize=dataSets.shape[0]
    diffMat= np.tile(inX,(dataSeteSize,1)) - dataSets
    sqDiffMAt=diffMat**2
    sqDistance=sqDiffMAt.sum(axis=1)
    distance=sqDistance**0.5
    sortedDistIndicies=distance.argsort()
    classCount={}

    for i in range(k):
        votelable=labels[sortedDistIndicies[i]]
        classCount[votelable]=classCount.get(votelable,0)+1
    sortedClassCount=sorted(classCount.iteritems(),
                            key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    groups,lables=creatTestData()
    draw(groups)
    s=classify0([0,0],groups,lables,2)
    print(s)