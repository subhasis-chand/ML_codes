import numpy as np

class PCA:
    def __init__(self, data, k):
        if type(data) is not np.ndarray and type(data) is not np.matrix:
            print("data must be numpy array")
            return
        if data.shape[1] <= k:
            print("k must be less than number of features")
            return
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

    def fit(self):
        eigVals, eigVecs = self.getEigVal()
        return eigVecs[:, 0 : self.k]

def main():
    m = np.matrix([[9,8,4], [9,6,8], [6,5,7], [3,4,7], [3,2,9]], dtype='float')
    pca = PCA(m, 2)
    print(pca.data)
    print(pca.getEigVal())
    print(pca.fit())

if __name__ == "__main__":
    main()
