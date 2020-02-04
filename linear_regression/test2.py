from linearRegressionAlternate import linearRegressionAlternate
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import linear_model
import numpy as np


def main():
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

    trainnigSetPercent = 75.0
    trainingIndex = int(x.shape[0] * trainnigSetPercent / 100.0)
    
    x_train = x[:trainingIndex , :]
    x_test = x[trainingIndex: , :]

    y_train = y[:trainingIndex , :]
    y_test = y[trainingIndex: , :]

    #using SKlearn lib linear regression
    linRegSkl = linear_model.LinearRegression()
    linRegSkl.fit(x_train, y_train)
    y_predSkl = linRegSkl.predict(x_test)

    #using own function for linear regression
    linReg = linearRegressionAlternate(x = x_train, y = y_train)
    theta = linReg.train()
    y_pred = linReg.test(x_test)

    fig = plt.figure()    
    ax = fig.subplots(1, 2)

    ax[0].plot(y_test, 'r.-')
    ax[0].plot(y_pred, 'g.-')
    ax[1].plot(y_test, 'r.-')
    ax[1].plot(y_predSkl, 'b.-')
    plt.show()

if __name__ == '__main__':
    main()






























