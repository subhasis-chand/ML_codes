import numpy as np

class PCA:
    def __init__(self, data, k):
        if type(data) is not np.ndarray and type(data) is not np.matrix:
            print("data must be numpy array")
            exit()
        if data.shape[1] <= k:
            print("k must be less than number of features")
            exit()
        self.data = np.matrix(data, dtype='float')
        self.k = k
        self.noOfRows = self.data.shape[0]
        self.noOfCols = self.data.shape[1]

    def covarianceMat(self):
        alpha = self.data - np.matrix(np.ones((self.noOfRows, self.noOfRows)), dtype='float') * self.data * (1.0/self.noOfRows)
        self.covMat = (alpha.T * alpha)/self.noOfRows
        return self.covMat

    def getEigVal(self):
        covMat = self.covarianceMat()
        return np.linalg.eig(covMat)

    def transform(self):
        eigVals, eigVecs = self.getEigVal()
        reducedEigVec = eigVecs[:, 0 : self.k]
        # m = np.matrix(np.zeros((self.noOfRows, self.k)))
        # transformedList = []
        # for i in range(self.data.shape[0]):
        #     m[i] = (reducedEigVec.T * self.data[i].T).T
        # return m
        return self.data * reducedEigVec

def main():
    import sklearn as sk
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from sklearn.datasets import load_breast_cancer
    from sklearn import decomposition
    rawData = load_breast_cancer()
    trainingData = rawData.data
    trainngLevels = rawData.target

    # Own PCA implementation
    # pca = PCA(trainingData, 3)
    # reducedData = pca.transform()
    # print("shape of reduced data", reducedData.shape)

    #sklearn PCA implementation
    pca = decomposition.PCA(n_components=3)
    pca.fit(trainingData)
    reducedData = pca.transform(trainingData)

    malignant, benign = [], []

    for i in range(trainngLevels.shape[0]):
        if trainngLevels[i] == 0:
            benign.append([reducedData[i, 0], reducedData[i, 1], reducedData[i, 2]])
        else:
            malignant.append([reducedData[i, 0], reducedData[i, 1], reducedData[i, 2]])

    benign = np.array(benign)
    malignant = np.array(malignant)
    
    plt.plot(benign[:, 0], benign[:, 1], '.')
    plt.plot(malignant[:, 0], malignant[:, 1], '.')
    plt.show()
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot3D(benign[:, 0], benign[:, 1], benign[:,2], '.')
    ax.plot3D(malignant[:, 0], malignant[:, 1], malignant[:, 2], '.')
    plt.show()
    
    


if __name__ == "__main__":
    main()
