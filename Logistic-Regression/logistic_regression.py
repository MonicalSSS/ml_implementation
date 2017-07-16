import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def costFunction(theta,X, y):
    m = X.shape[0]
    J = 0
    z = X.dot(theta)
    J = (-y.T.dot(np.log(sigmoid(z))) - (1 - y.T).dot(np.log(1 - sigmoid(z)))) / m
    return J[0]


def gradient(theta,X, y): 
    m = X.shape[0] #X.shape = (100,3)
    grad = np.zeros_like(theta)
    z = X.dot(theta.reshape(-1,1)) # theta.shape = (3,) theta.reshape(-1,1).shape = (3,1)
    grad = (1.0 / m) * X.T.dot(( sigmoid(z) - y )) 
    # grad =  X.T.dot(( sigmoid(z) - y )) 
    return grad.flatten()

