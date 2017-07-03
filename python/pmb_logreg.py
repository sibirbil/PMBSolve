#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 21:13:34 2017

@author: kaan
"""

# NOTES:
# It's hard to get the logistic regression going. Usually the theta*X grows
# too large so that exp overflows. The sigmoid h becomes 1, and log(h) becomes 0.
# We rewrite the objective function such that we avoid a zero argument to log.
# Also, we rescale the input by the range of the data.

# The MNIST data files in CSV form are downloaded from:
# https://pjreddie.com/projects/mnist-in-csv/

# The files exceed GitHub's file size limit. Please download them to your
# computer separately to run the code.

import numpy as np
from pmb_scipy import pmbsolve
from scipy.optimize import minimize

def sigmoid(z):
    return 1/(1+np.exp(-z))

def logreg(theta, *args):
    # args = (X,y,lambda)
    m = args[0].shape[0]
    lam = args[2] # regularization parameter
    z = np.dot(args[0],theta)
    J = sum(z + np.log(1+np.exp(-z))) - np.dot(z,args[1])
    J += 0.5*lam*np.dot(theta[1:],theta[1:])
    return J/m

def logreg_deriv(theta, *args):
    # args = (X,y,lambda)
    lam = args[2]
    z = np.dot(args[0],theta)
    grad = np.dot(args[0].transpose(), 1/(1+np.exp(-z)) - y)
    grad[1:] += lam * theta[1:]
    return grad/args[0].shape[0]

train = np.loadtxt("data/mnist_train.csv",delimiter=",")

X = np.copy(train)/train.max()
X[:,0] = np.ones(X.shape[0])
value = 5
y = np.array([1 if target==value else 0 for target in train[:,0]])

m, n = X.shape
theta_0 = np.random.randn(n)

#res = minimize(logreg, theta_0, args=(X,y,1), jac=logreg_deriv, method="L-BFGS-B")
res = minimize(logreg, theta_0, args=(X,y,1), jac=logreg_deriv, method=pmbsolve)

# Test the predictor
# Load the test data (not included in Github -- see the note at the beginning)
test = np.loadtxt("data/mnist_test.csv",delimiter=",")
Xtest = np.copy(test)/test.max()
Xtest[:,0] = np.ones(Xtest.shape[0])

ytest = np.array([1 if target==value else 0 for target in test[:,0]])
pred = np.array([1 if sigmoid(np.dot(xt, res.x))>0.5 else 0 for xt in Xtest])

tp = np.sum(np.logical_and(ytest==1,pred==1))
fn = np.sum(np.logical_and(ytest==1,pred==0))
fp = np.sum(np.logical_and(ytest==0,pred==1))
tn = np.sum(np.logical_and(ytest==0,pred==0))

print("Accuracy =", (tp+tn)/(tp+tn+fp+fn))