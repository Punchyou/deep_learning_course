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
from testCases_v2 import *

np.random.seed(1) # set s seed for consistent results

# load the dataset: X: matrix of features, Y: matrix of labels
X, Y= load_planar_dataset()


colors = ["purple" if i == 0 else "orange" for i in Y[0]]
# Visualize the data
plt.scatter(X[0, :], X[1, :], c=colors)

# understand the shape of the data
shape_X =   X.shape
shape_Y = Y.shape
# number of examples
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
    Returns the input, hidden and output layers sizes.
    """

    # input layer size
    n_x = len(X)
    n_h = 4
    n_y = len(Y)
    
    return n_x, n_h, n_y

def initialize_parameters(n_x, n_h, n_y):
    """
    Returns the parameters:
        W1: weight matrix of shape (n_h, n_x)
        b1: bias vector of shape (n_h, 1)
        W2: weight matrix of shape (n_y, n_h)
        b2: bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) # for consistent results
    
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))
    
    assert (W1.shape == (n_h, n_x)) * 0.01
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h)) * 0.01
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# loor the forward propagation
def forward_propagation(X, parameters):
    """
    Returns the sigmoid A2 of the second activation and
    cache - a dict containing Z1, A1, Z2, A2.
    """
    
    #parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    
     # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = 1/(1 + np.exp(-Z2)) #sigmoid
    assert(A2.shape == (1, X.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache

X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x, n_h, n_y)
X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)
print(np.mean(cache['Z1']), np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))













