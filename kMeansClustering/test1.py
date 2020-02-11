import numpy as np
from kMeansCLustering import kMeans

def generateThreeClusters2dData():
    #generate a sample 2d data
    data = np.zeros((90, 2))

    for i in range(90):
        if i < 30:
            x_c, y_c = 1.0, 1.0
        elif i >= 30 and i < 60:
            x_c, y_c = 3.0, 3.0
        else:
            x_c, y_c = 4.0, 2.0
        data[i] = np.array([ x_c + np.random.rand(), y_c + np.random.rand()])
    
    return np.matrix(data)

def main():
    data = generateThreeClusters2dData()
    kmeans = kMeans(data)
    y = kmeans.run(animation=True, printClusterCenter=True, printLoss=True)
    
    

if __name__ == "__main__":
    main()