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
        prevLoss = self.loss()
        bestClusterCenters = np.copy(self.clusterCenters)
        bestY = np.copy(self.y)
        leastLoss = self.loss()

        if animation:
            fig = plt.figure()    
            ax = fig.subplots()

        for ite in range(noOfIter):
            print("No of iteration: ", ite+1)
            self.randomInitClusterCenters(k)
            prevLoss = self.loss()
            while True:
                self.assignCluster()
                self.moveClusterCenters()
                currentLoss = self.loss()
                if printLoss:
                    print("Loss is: ", self.loss())
                if printClusterCenter:
                    print("cluster center is", self.clusterCenters)
                if animation and self.n == 2:
                    plt.cla()
                    ax.plot(self.clusterCenters[:, 0], self.clusterCenters[:, 1], 'x')
                    for i in range(self.clusterCenters.shape[0]):
                        clusteredData = self.x[self.y == i]
                        ax.plot(clusteredData[:, 0], clusteredData[:, 1], '.')
                    title = "Iteration: " + str(ite+1)
                    ax.set_title(title)
                    plt.pause(0.2)
                if animation and self.n != 2:
                    print("To show the animation, no of features should be equal to two. \n \
                        Otherwise how do you expect to visualise a multi-dimensional data on a 2D display?")
                if abs(currentLoss - prevLoss) < threshold:
                    break
                else:
                    prevLoss = currentLoss

            if currentLoss < leastLoss:
                bestClusterCenters = np.copy(self.clusterCenters)
                bestY = np.copy(self.y)
                leastLoss = currentLoss

        self.clusterCenters = np.copy(bestClusterCenters)
        self.y = np.copy(bestY)
        self.loss = leastLoss
            
        if animation:
            plt.close()
            fig = plt.figure()    
            ax = fig.subplots()
            ax.plot(self.clusterCenters[:, 0], self.clusterCenters[:, 1], 'x')
            for i in range(self.clusterCenters.shape[0]):
                clusteredData = self.x[self.y == i]
                ax.plot(clusteredData[:, 0], clusteredData[:, 1], '.')
            ax.set_title("The best possible clustering out of all the iteration")
            plt.show()
        return self.y

