import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

class linearRegressionAlternate:
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

        x0 = np.matrix(np.ones((x.shape[0], 1)))
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

    def train(self, printTheta=False):
        self.theta = np.linalg.inv(self.x.T * self.x) * self.x.T * self.y
        return self.theta

    def test(self, x_test):
        x0 = np.matrix(np.ones((x_test.shape[0], 1)))
        x_test = np.hstack((x0, x_test))
        return x_test * self.theta
        
