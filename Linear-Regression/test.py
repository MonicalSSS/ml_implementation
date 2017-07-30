#!/usr/bin/env python
from linear_regression import *
import pandas as pd 

def loadData():
    data = pd.read_csv('housing.csv')
    prices = data['MEDV']
    features = data.drop('MEDV', axis = 1)
    X = features.get_values()
    y = data.get_values()
    print X[:,0]
    print y
    return X[:,0],y.astype(int)

def plotScatter(X,y):
    #plot scatter image
    #colors = ['b', 'c', 'y', 'm', 'r']
    plt.plot(X,y,'o',color='b')
    # plt.ylabel('profit in $10,000s')
    # plt.xlabel('Population of City in 10,000s')
    plt.show()


if __name__ == "__main__":
    
    X,y = loadData()
    m = len(X)
    print m
    plotScatter(X,y)

    # X = np.reshape(X, (m, 1))
    # print (type(X), X.shape)
    # print (type(y), y.shape)

    # X = np.c_[np.ones(m), X] # add one col
    # theta = np.zeros(2)
    # iterations = 1500
    # alpha = 0.01

    # linearRegression(X,y,theta,alpha,iterations)

    