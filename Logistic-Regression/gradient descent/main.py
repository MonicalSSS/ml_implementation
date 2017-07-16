import numpy as np
import matplotlib.pyplot as plt
import os


def loadData():
    path = os.getcwd() + '/testSet.txt'
    data = np.loadtxt(path)
    # path = os.getcwd() + '/ex2data1.txt'
    # data = np.loadtxt(path,delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    return X, y


def scatterPlot():
    X, y = loadData()
    y = y.astype(int)
    pos = (y == 1)
    neg = (y == 0)
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], c='red')
    plt.scatter(X[neg, 0], X[neg, 1], c='b')
    plt.show()


def decisionBoundary(theta):
    X, y = loadData()
    y = y.astype(int)
    pos = (y == 1)
    neg = (y == 0)
    fig = plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], c='red')
    plt.scatter(X[neg, 0], X[neg, 1], c='b')
    boundary_xs = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
    boundary_ys = (-1. / theta[2]) * (theta[0] + theta[1] * boundary_xs)
    plt.plot(boundary_xs, boundary_ys)
    plt.show()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def costFunction(X, y, theta):
    m = X.shape[0]
    J = 0
    z = X.dot(theta)
    J = (-y.T.dot(np.log(sigmoid(z))) - (1 - y.T).dot(np.log(1 - sigmoid(z)))) / m
    return J[0]


def gradientFunction(X, y, theta):
    m = X.shape[0]
    grad = np.zeros_like(theta)
    z = X.dot(theta)
    grad = X.T.dot(sigmoid(z) - y.reshape(-1, 1)) / m 
    return grad


def logisticRegression(X, y, alpha, maxIters):
    m, n = X.shape
    # theta = np.ones((n,1))
    theta = np.zeros((n, 1))
    z = X.dot(theta)
    for iters in range(maxIters):
        grad = gradientFunction(X, y, theta)
        theta -= alpha * grad
    return theta


if __name__ == '__main__':
    X, y = loadData()
    scatterPlot()
    X = np.c_[np.ones((X.shape[0], 1)), X]
    initial_theta = np.zeros((X.shape[1], 1))
    cost = costFunction(X, y, initial_theta)
    grad = gradientFunction(X, y, initial_theta)

    print cost
    print grad
    alpha = 0.1
    maxIters = 400
    theta = logisticRegression(X, y, alpha, maxIters)
    print (theta)
    decisionBoundary(theta)
