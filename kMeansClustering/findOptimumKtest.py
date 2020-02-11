import numpy as np
import matplotlib.pyplot as plt
from kMeansCLustering import kMeans
from test2 import generateFourClusters2dData
from test1 import generateThreeClusters2dData

def testForThreeClusters(K):
    data = generateThreeClusters2dData()
    kmeans = kMeans(data)
    losses = kmeans.findOptimumKvalue(K=K)

    fig = plt.figure()    
    ax = fig.subplots(2)
    ax[0].plot(data[:, 0], data[:, 1], '.')
    ax[0].set_title("Elbow point for data with Three proper clusters")
    ax[1].plot([i for i in range(2, K+1)] ,losses, '.-')
    plt.show()
#####################################################################

def testForFourClusters(K):
    data = generateFourClusters2dData()
    kmeans = kMeans(data)
    losses = kmeans.findOptimumKvalue(K=K)

    fig = plt.figure()    
    ax = fig.subplots(2)
    ax[0].plot(data[:, 0], data[:, 1], '.')
    ax[0].set_title("Elbow point for data with Four proper clusters")
    ax[1].plot([i for i in range(2, K+1)] ,losses, '.-')
    plt.show()

#####################################################################

def testForNoProperCluster(K):
    height = []
    for i in range(100):
        for j in range(3):
            height.append(float(145 + i))

    weight = []
    for i in range(len(height)):
        weight.append(height[i]/2.0 - 10 * np.random.rand())

    height = np.matrix(height)
    weight = np.matrix(weight)
    data = np.hstack((height.T, weight.T))

    kmeans = kMeans(data)
    losses = kmeans.findOptimumKvalue(K=K)

    fig = plt.figure()    
    ax = fig.subplots(2)
    ax[0].plot(data[:, 0], data[:, 1], '.')
    ax[0].set_title("Elbow point for data with Three proper clusters")
    ax[1].plot([i for i in range(2, K+1)] ,losses, '.-')
    plt.show()

def testForAllClusters(K):
    print("Testing for Three clusters")
    testForThreeClusters(K)
    print("Testing for Four clusters")
    testForFourClusters(K)
    print("Testing for improper clusters")
    testForNoProperCluster(K)


if __name__ == "__main__":
    testForAllClusters(7)