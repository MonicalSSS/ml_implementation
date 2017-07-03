#!/usr/bin/env python
from linear_regression import *

def loadData():
    file1 = open("ex1data1.txt", "r")
    file2 = open("ex1data2.txt", "r")
    X, y = np.loadtxt(file1, dtype=float, delimiter=",", unpack=True)
    return X,y

def plotScatter(X,y):
    #plot scatter image
    #colors = ['b', 'c', 'y', 'm', 'r']
    plt.plot(X,y,'x',color='r')
    plt.ylabel('profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.show()


if __name__ == "__main__":
    
    X,y = loadData()
    m = len(X)
    plotScatter(X,y)

    X = np.reshape(X, (m, 1))
    print (type(X), X.shape)
    print (type(y), y.shape)

    X = np.c_[np.ones(m), X] # add one col
    theta = np.zeros(2)
    iterations = 1500
    alpha = 0.01

    linearRegression(X,y,theta,alpha,iterations)

    