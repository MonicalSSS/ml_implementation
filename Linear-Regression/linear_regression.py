import numpy as np
import matplotlib.pyplot as plt


def computeCost(X, y, theta):
    m = len(y)
    cost = sum((np.dot(X, theta) - y)**2) / (2 * m)
    return cost


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
        Z = X.T.dot(X.dot(theta) - y) * alpha / m
        theta = theta - Z
        J_history[iter] = computeCost(X,y,theta) 
    return theta,J_history

def linearRegression(X,y,theta,alpha,num_iters):
    theta ,J_history = gradientDescent(X,y,theta,alpha,num_iters)
    plt.plot(X[:,1],y,'x',color='r')
    plt.ylabel('profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.plot(X[:,1],X.dot(theta),'-',color='b')
    plt.show()
