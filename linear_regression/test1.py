from linearRegression import LinearRegression
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import linear_model
import numpy as np

def main1():
    # admissionData = genfromtxt("../resources/graduate-admissions/Admission_Predict_Ver1.1.csv", delimiter=',')
    admissionData = genfromtxt("../resources/graduate-admissions/Admission_Predict.csv", delimiter=',')
    x = np.matrix(admissionData[1:, 1:-1])
    # Normalising the data
    x[:, 0] = x[:, 0]/340.0
    x[:, 1] = x[:, 1]/120.0
    x[:, 2] = x[:, 2]/5.0
    x[:, 3] = x[:, 3]/5.0
    x[:, 4] = x[:, 4]/5.0
    x[:, 5] = x[:, 5]/10.0

    y = np.matrix(admissionData[1:, -1:])

    # Spliting the data for training and testing
    trainnigSetPercent = 75.0
    trainingIndex = int(x.shape[0] * trainnigSetPercent / 100.0)
    
    x_train = x[:trainingIndex , :]
    x_test = x[trainingIndex: , :]

    y_train = y[:trainingIndex , :]
    y_test = y[trainingIndex: , :]

    linReg = LinearRegression(x = x_train, y = y_train)
    theta = linReg.train(animation=True, thresHold=0.00005, alpha=0.01)
    y_pred = linReg.test(x_test)

    plt.plot(y_test, 'r.-')
    plt.plot(y_pred, 'g.-')
    plt.show()

if __name__ == '__main__':
    main1()





























