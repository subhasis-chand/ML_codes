import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, x=np.zeros(5), y=np.zeros(5)):
        if type(x) is not np.matrix:
            print("input x must be numpy matrix")
            exit()
        if type(y) is not np.matrix:
            print("output y must be numpy matrix")
            exit()
        if x.shape[0] != y.shape[0]:
            print("no of training examples for input and out put must be the same")
            exit()

        x0 = np.matrix(np.ones((x.shape[0], 1)))
        self.x = np.hstack((x0, x))
        self.y = y
        self.m = self.x.shape[0]
        self.n = self.x.shape[1]
        self.theta = np.matrix(np.ones((self.n, 1)))

    def hypothesis(self, x):
        return self.theta.T * x   #x is a vector [nx1]

    def loss(self):
        return np.sum(np.square(self.x * self.theta - self.y)) / (2.0 * self.m)

    def gradientDescent(self, animation=False, printLoss=False, printTheta=False, thresHold=0.001, alpha=0.01 ):
        fig = plt.figure()    
        ax = fig.subplots(1, 2)

        if animation:
            ax[0].yaxis.grid(color='gray', linestyle='dashed')
            ax[0].xaxis.grid(color='gray', linestyle='dashed')

        theta = np.copy(self.theta)
        lossArr = []
        ite = 0
        while True:
            ite += 1

            predictDiff = self.x * self.theta - self.y
            self.theta = self.theta - alpha / (float(self.m)) * (self.x.T * predictDiff)

            actualLoss = self.loss()
            lossArr.append(actualLoss)

            if animation:
                plt.cla()
                plt.grid()
                if len(lossArr) >= 2:
                    diffInLoss = abs(lossArr[-1] - lossArr[-2])
                else:
                    diffInLoss = 0
                title0 = "Iteration: " + str(ite) + "    " + "Loss: " + str(round(actualLoss, 5)) \
                    + "    " + "Diff in Loss: " + str(round(diffInLoss, 5))
                ax[0].set_title(title0)
                ax[0].set_axisbelow(True)
                ax[0].plot(lossArr, '.r')

                if self.n == 2:
                    title1 = "theta0: " + str(round(self.theta[0, 0], 5)) + "    theta1: " + str(round(self.theta[1, 0], 5))
                    ax[1].set_title(title1) 
                    ax[1].plot([-1, 16],[-1, 16], 'x')
                    ax[1].plot(self.x[:, 1], self.y[:, 0], '.g')
                    ax[1].plot([self.x[0, 1], self.x[-1, -1]], [self.hypothesis(self.x[0, :].T)[0, 0], self.hypothesis(self.x[-1, :].T)[0, 0]], 'r')

                plt.pause(0.3)

            if printTheta:
                print("theta: ", theta)
            if printLoss:
                print("loss: ", self.loss())

            if len(lossArr) >= 2 and abs(lossArr[-1] - lossArr[-2]) < thresHold:
                if animation:
                    plt.cla()
                    plt.grid()
                    title = "Iteration: " + str(ite) + "    " + "Loss: " + str(round(actualLoss, 5))
                    ax[0].set_title(title)
                    ax[0].set_axisbelow(True)
                    plt.plot([0,10], [0,10], 'x')
                    plt.plot(lossArr, '.')
                    plt.show()
                return self.theta

    def train(self, animation=False, printLoss=False, printTheta=False, thresHold=0.001, alpha=0.01):
        return self.gradientDescent(animation, printLoss, printTheta, thresHold, alpha)

    def test(self, x_test):
        x0 = np.matrix(np.ones((x_test.shape[0], 1)))
        x_test = np.hstack((x0, x_test))
        return x_test * self.theta






