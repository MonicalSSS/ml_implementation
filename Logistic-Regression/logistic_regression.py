import numpy as np
import matplotlib.pyplot as plt 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def costFunction(X,y,theta):
    m = len(y)
    J = 0
    grad = np.zeros_like(theta)
    z = X.dot(theta)