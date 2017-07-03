import numpy as np
import matplotlib.pyplot as plt 

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def costFunction(X,y,theta):
    m = len(y.T)
    J = 0
    grad = np.zeros_like(theta)
    z = X.dot(theta)
    J = (-y.dot(np.log(sigmoid(z))) - (1 - y).dot(np.log(1 - sigmoid(z)))) / m
    grad = (X.T.dot((sigmoid(z) - y.T))/m)
    return J ,grad 
