## K-means lcustering

This code base written from scratch to properly understand how k-means clustering algorithm works.
For 2D data set the working for the algorithm can be visualised step by step.

* kMeansCLustering.py : This script has the code to create an object for k-means. You can run the algorithm on your data, check for elbow point to figure out best possible value of K using this code.
* test1.py : Generates a random 2D data set and tests the data for kmenas. The animation option here is set to True. So every step which is occuring in the process can be visible.
* test2.py: It does all the above operations for Four clusters
* In the above two examples we know how many clusters are there. So it is easy to set a K value to run the algo.
* tshirtSizing.py : We created a random data set for height vs weight for 300 people. The goal is to cluster the data to determine various t-shirt sizes to be produced. This code runs k-means with k=3, 4 and 5 with 100 iterations in each k. Finally shows up the clustering in one colorful chart with respective cluster centers.
* findOptimumKvalue.py : This code demonstrates how to draw an elbow plot and determine the best k for your data. It runs k = 2 to 7 (you can set it in the code) for three types of data sets.

Download ans check the .png files and the .mp4 file to see the results. 