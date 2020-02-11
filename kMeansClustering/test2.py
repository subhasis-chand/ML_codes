import numpy as np
from kMeansCLustering import kMeans

def generateFourClusters2dData():
    #generate a sample 2d data
    data = np.zeros((200, 2))

    for i in range(data.shape[0]):
        if i < 50:
            x_c, y_c = 1.0, 1.0
        elif i >= 50 and i < 100:
            x_c, y_c = 3.0, 3.0
        elif i >= 100 and i < 150:
            x_c, y_c = 5.0, 4.0
        else:
            x_c, y_c = 4.0, 2.0
        data[i] = np.array([ x_c + np.random.rand(), y_c + np.random.rand()])
    
    return np.matrix(data)

def main():
    import time
    time.sleep(2)
    data = generateFourClusters2dData()
    kmeans = kMeans(data)
    y = kmeans.run(k=4, animation=True, printClusterCenter=True, printLoss=True)
    
    

if __name__ == "__main__":
    main()