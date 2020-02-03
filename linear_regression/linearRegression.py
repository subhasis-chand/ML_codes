import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import linear_model

class LinearRegression:
    def __init__(self, x=np.zeros(5), y=np.zeros(5)):
        if type(x) is not np.matrix:
            print("input x must be numpy matrix")
            exit()
        if type(y) is not np.matrix:
            print("output y must be numpy matrix")
            exit()
        if x.shape[0] != x.shape[0]:
            print("no of training examples for input and out put must be the same")
            exit()
        # print("x: ", x)
        # print("y: ", y)
        # exit()

        x0 = np.matrix(np.ones((x.shape[0], 1)))
        # x = x - np.matrix(np.ones((x.shape[0], x.shape[0]))) * x * (1.0/x.shape[0]) 
        # y = y - np.matrix(np.ones((y.shape[0], y.shape[0]))) * y * (1.0/y.shape[0]) 
        # x = x / x.max()
        # y = y / y.max()
        self.x = np.hstack((x0, x))
        self.y = y
        self.m = self.x.shape[0]
        self.n = self.x.shape[1]
        self.theta = np.matrix(np.ones((self.n, 1)))

    def hypothesis(self, x):
        return self.theta.T * x   #x is a vector

    def loss(self):
        l = 0.0
        for i in range(self.m):
            row = self.x[i, :].T
            l += (self.hypothesis(row) - self.y[i, 0]) ** 2
        return l / (2.0 * self.m)

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
            for i in range(self.n):
                l = 0
                for j in range(self.m):
                    row = self.x[j, :].T
                    l += (self.hypothesis(row) - self.y[j, 0]) * self.x[j, i] 
                theta[i, 0] = self.theta[i, 0] - alpha * l / (float(self.m))
            self.theta = np.copy(theta)
            actualLoss = self.loss()[0,0]
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
                # ax[0].plot([0,10], [0,10], 'x')
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






def main():
    # x = np.arange(1,15,0.2)
    # y = 2 + np.random.rand() * np.random.rand(x.shape[0]) * 20
    # y.sort()
    # np.savetxt('y_for_lin_reg_one_var.txt', y)
    # np.savetxt('x_for_lin_reg_one_var.txt', x)

    x = np.loadtxt("../resources/x_for_lin_reg_one_var.txt")
    y = np.loadtxt("../resources/y_for_lin_reg_one_var.txt")


    linReg = LinearRegression(np.matrix(x).T, np.matrix(y).T, 0.001)
    print(linReg.loss())
    theta = linReg.gradientDescent(animation=True, thresHold=0.0001)
    print(theta)
    print(linReg.loss())

def main1():
    # admissionData = genfromtxt("../resources/graduate-admissions/Admission_Predict_Ver1.1.csv", delimiter=',')
    admissionData = genfromtxt("../resources/graduate-admissions/Admission_Predict.csv", delimiter=',')
    x = np.matrix(admissionData[1:, 1:-1])
    x[:, 0] = x[:, 0]/340.0
    x[:, 1] = x[:, 1]/120.0
    x[:, 2] = x[:, 2]/5.0
    x[:, 3] = x[:, 3]/5.0
    x[:, 4] = x[:, 4]/5.0
    x[:, 5] = x[:, 5]/10.0

    y = np.matrix(admissionData[1:, -1:])

    trainnigSetPercent = 75.0
    trainingIndex = int(x.shape[0] * trainnigSetPercent / 100.0)
    
    x_train = x[:trainingIndex , :]
    x_test = x[trainingIndex: , :]

    y_train = y[:trainingIndex , :]
    y_test = y[trainingIndex: , :]

    #with sklearn
    linReg = linear_model.LinearRegression()
    linReg.fit(x_train, y_train)
    y_pred = linReg.predict(x_test)

    #with own algo
    # linReg = LinearRegression(x = x_train, y = y_train)
    # theta = linReg.train(animation=True, thresHold=0.0001, alpha=0.01)
    # y_pred = linReg.test(x_test)

    print(y_pred)
    print(y_test)
    plt.plot(y_test, 'r')
    plt.plot(y_pred, 'g')
    plt.show()

if __name__ == '__main__':
    main1()





























