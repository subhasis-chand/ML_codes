import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, x=np.zeros(5), y=np.zeros(5), alpha = 1):
        if type(x) is not np.matrix:
            print("input x must be numpy matrix")
            return
        if type(y) is not np.matrix:
            print("output y must be numpy matrix")
            return
        if x.shape[0] != x.shape[0]:
            print("no of training examples for input and out put must be the same")
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
        self.alpha = alpha
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

    def gradientDescent(self, animation=False, printLoss=False, printTheta=False, thresHold=0.001 ):
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
                theta[i, 0] = self.theta[i, 0] - self.alpha * l / (float(self.m))
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
                title0 = "Me lodu hu: " + str(ite) + "    " + "Loss: " + str(round(actualLoss, 5)) \
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



if __name__ == '__main__':
    main()





























