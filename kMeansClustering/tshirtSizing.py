import numpy as np
import matplotlib.pyplot as plt
from kMeansCLustering import kMeans

height = []
for i in range(100):
    for j in range(3):
        height.append(float(145 + i))

weight = []
for i in range(len(height)):
    weight.append(height[i]/2.0 - 10 * np.random.rand())

height = np.matrix(height)
weight = np.matrix(weight)
data = np.hstack((height.T, weight.T))

max_vals = data.max(axis=0)

plt.close()
fig = plt.figure()    
ax = fig.subplots(3)

kmeans = kMeans(data)
kmeans.run(k=3, noOfIter=100)
for i in range(kmeans.clusterCenters.shape[0]):
    clusteredData = data[kmeans.y == i]
    ax[0].plot(clusteredData[:, 0], clusteredData[:, 1], '.')
ax[0].plot(kmeans.clusterCenters[:, 0] * max_vals[0, 0], kmeans.clusterCenters[:, 1] * max_vals[0, 1], 'xk')
ax[0].set_title("clustering with Three, Four and Five t-Shirt sizes respectively")
ax[0].set_xlabel("Height in cms")
ax[0].set_ylabel("Weight in Kgs")


kmeans = kMeans(data)
kmeans.run(k=4, noOfIter=100)
for i in range(kmeans.clusterCenters.shape[0]):
    clusteredData = data[kmeans.y == i]
    ax[1].plot(clusteredData[:, 0], clusteredData[:, 1], '.')
ax[1].plot(kmeans.clusterCenters[:, 0] * max_vals[0, 0], kmeans.clusterCenters[:, 1] * max_vals[0, 1], 'xk')
ax[1].set_xlabel("Height in cms")
ax[1].set_ylabel("Weight in Kgs")

kmeans = kMeans(data)
kmeans.run(k=5, noOfIter=100)
for i in range(kmeans.clusterCenters.shape[0]):
    clusteredData = data[kmeans.y == i]
    ax[2].plot(clusteredData[:, 0], clusteredData[:, 1], '.')
ax[2].plot(kmeans.clusterCenters[:, 0] * max_vals[0, 0], kmeans.clusterCenters[:, 1] * max_vals[0, 1], 'xk')
ax[2].set_xlabel("Height in cms")
ax[2].set_ylabel("Weight in Kgs")

plt.show()
