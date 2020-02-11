import numpy as np
import matplotlib.pyplot as plt

class kMeans():
    def __init__(self, x=np.zeros(5)):
        if type(x) is not np.matrix:
            print("input x must be numpy matrix")
            exit()
        
        self.x = x / x.max(axis=0)
        self.m = self.x.shape[0]
        self.n = self.x.shape[1]
        self.y = np.zeros(self.m)
    
    def randomInitClusterCenters(self, k):
        clusterCenters = []
        for i in range(k):
            clusterCenters.append(np.ones(self.n) * np.random.rand())
        self.clusterCenters = np.matrix(clusterCenters)

    def assignCluster(self):
        dist = -1
        for i in range(self.m):
            self.y[i] = np.argmin(np.linalg.norm(self.x[i] - self.clusterCenters, axis=1))

    def moveClusterCenters(self):
        for i in range(self.clusterCenters.shape[0]):
            self.clusterCenters[i] = np.mean(self.x[self.y == i], axis=0)
    
    def loss(self):
        self.totalLoss = 0
        for i in range(self.clusterCenters.shape[0]):
            x = self.x[self.y == i]
            if x.shape[0] != 0:
                self.totalLoss += np.linalg.norm(x - self.clusterCenters[i])
        return self.totalLoss


    def run(self, k = 3, threshold=0.01, animation=False, noOfIter=30, printClusterCenter=False, printLoss=False):
        self.randomInitClusterCenters(k)
        for i in range(noOfIter):
            self.assignCluster()
            self.moveClusterCenters()
            if printLoss:
                print("Loss is: ", self.loss())
            if printClusterCenter:
                print(self.clusterCenters)
            if animation and self.n == 2:
                plt.cla()
                plt.plot(self.clusterCenters[:, 0], self.clusterCenters[:, 1], 'x')
                for i in range(self.clusterCenters.shape[0]):
                    clusteredData = self.x[self.y == i]
                    plt.plot(clusteredData[:, 0], clusteredData[:, 1], '.')
                plt.pause(0.2)
            if animation and self.n != 2:
                print("To show the animation, no of features should be equal to two")
        plt.show()
        return self.y



def generateThreeClusters2dData():
    #generate a sample 2d data
    data = np.zeros((90, 2))

    for i in range(90):
        if i < 30:
            x_c, y_c = 1.0, 1.0
        elif i >= 30 and i < 60:
            x_c, y_c = 3.0, 3.0
        else:
            x_c, y_c = 4.0, 2.0
        data[i] = np.array([ x_c + np.random.rand(), y_c + np.random.rand()])
    
    return np.matrix(data)

def main():
    data = generateThreeClusters2dData()
    kmeans = kMeans(data)
    kmeans.run(animation=True, printClusterCenter=True, printLoss=True)

if __name__ == "__main__":
    main()