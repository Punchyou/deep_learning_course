# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
We are building a model - classifier - that can define regions as red or blue."""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model as slm
from planar_utils import *

np.random.seed(1) # set s seed for consistent results

# load the dataset: X: matrix of features, Y: matrix of labels
X, Y= load_planar_dataset()


colors = ["purple" if i == 0 else "orange" for i in Y[0]]
# Visualize the data
plt.scatter(X[0, :], X[1, :], c=colors)

# understand the shape of the data
shape_X =   X.shape
shape_Y = Y.shape
# number of exmples
m = X.shape[1]

#logistic regression approach
clf = slm.LogisticRegressionCV()
clf.fit(X.T, Y.T) #why transpose?

# plot it
plot_decision_boundary(lambda x: clf.predict(x), X, Y) #why lambda here?
plt.title("Logistic Regression")

#print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")


# neural network approach
def layer_sizes(X, Y):
    """
    Returns the imput, hidden and output layers sizes
    """

    # input layer size
    n_x = len(X[0])
    n_h = 4
    n_y = len(Y[0])