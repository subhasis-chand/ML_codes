import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
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
        self.y_pred = None

    def hypothesis(self, x=None):
        if x is None:
            x = self.x
        else:
            x0 = np.matrix(np.ones((x.shape[0], 1)))
            x = np.hstack((x0, x))
        return 1.0 / (1.0 + np.power(np.e, ((-1.0) * x * self.theta))) 

    def loss(self):
        return (-1.0 / self.m) * np.sum(np.multiply(self.y, np.log(self.hypothesis())) \
            + np.multiply((1.0 - self.y), np.log(1 - self.hypothesis())))

    def gradientDescent(self, animation=False, printLoss=False, printTheta=False, thresHold=0.001, alpha=0.01):
        self.theta = self.theta - alpha * (self.x.T * (self.hypothesis() - self.y)) # This is correct. Don't get confused
        prevLoss = self.loss()
        ite = 1
        lossArr = [prevLoss]
        fig = plt.figure()    
        ax = fig.subplots()
        while True:
            ite += 1
            self.theta = self.theta - alpha * (self.x.T * (self.hypothesis() - self.y))
            currentLoss = self.loss()
            lossArr.append(currentLoss)

            if animation:
                plt.cla()
                ax.plot(lossArr, '.')
                title = "Iteration: " + str(ite) + "    " + "Loss: " + str(round(currentLoss, 5)) \
                    + "    " + "Diff in Loss: " + str(round(abs(prevLoss - currentLoss), 5))
                ax.set_title(title)
                plt.pause(0.3)

            if printTheta:
                print("Theta: ", self.theta)
            if printLoss:
                print("Loss: ", currentLoss)

            if abs(prevLoss - currentLoss) < thresHold:
                break
            prevLoss = currentLoss
        if animation:
            plt.show()

    def train(self, animation=False, printLoss=False, printTheta=False, thresHold=0.001, alpha=0.01):
        return self.gradientDescent(animation, printLoss, printTheta, thresHold, alpha)

    def test(self, x):
        y_pred = self.hypothesis(x)
        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred >= 0.5] = 1
        self.y_pred = y_pred
        return y_pred

    def scores(self, y_test):
        if y_test.shape != self.y_pred.shape:
            print("shape of y_test ", y_test.shape, "did not match shape of y_pred ", self.y_pred)
        tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
        confusionMatrix = np.matrix(np.zeros((2,2)))
        for i in range(len(y_test)):
            if y_test[i, 0] == 0 and self.y_pred[i, 0] == 0:
                tn += 1
            elif y_test[i, 0] == 0 and self.y_pred[i, 0] == 1:
                fp += 1
            elif y_test[i, 0] == 1 and self.y_pred[i, 0] == 0:
                fn += 1
            else:
                tp += 1
        confusionMatrix[0, 0] = tn
        confusionMatrix[0, 1] = fp
        confusionMatrix[1, 0] = fn
        confusionMatrix[1, 1] = tp

        precision = tp / (tp + tn)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return { 'precision': precision, 'recall': recall, 'f1': f1, 'accuaracy': accuracy, 'confusionMatrix':confusionMatrix }

