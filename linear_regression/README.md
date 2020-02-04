## Linear Regression

This code base written from scratch for understanding of linear regression model.
The algorithm is present in following files
* linearRegression.py
* linearRegressionAlternate.py

The first one uses gradient descent with animation option. By setting the animation option to true the loss and number of iterations are plotted in every iteration which gives a awesome intutition about how the model is learning.

The Second file uses the Normal Equation to find out the parameters without gradient descent and gives a best fit.

There are three testing files.
* demoAndVisLinReg.py : This file runs linear regression on a one dimensional data set from resources dir. It nicely shows how the hypothesis fits the data set when loss changes with gradient descent.
* test1.py: It runs linear regression on  a [dataset](https://www.kaggle.com/mohansacharya/graduate-admissions) and plots the predicted values along with the actual values
* test2.py: It runs on the above data set using normal equation and also runs lin reg from SKlearn. And shows the comparision.

Check the images to see the final results.