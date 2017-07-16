from logistic_regression import *
import scipy.optimize as op
import os

def loadData():
    # path  = os.getcwd() + '/ex2data1.txt'
    # data = np.loadtxt(path,dtype=float,delimiter=',')
    path = os.getcwd() + '/testSet.txt'
    data = np.loadtxt(path)
    X = data[:,0:2]
    y = data[:,2]
    return X,y

def plotScatter():
    X,y = loadData()
    y = y.astype(int)
    neg = (y == 0)
    pos = (y == 1)
    lable1 = plt.scatter(X[neg,0],X[neg,1],marker='o',c='b')
    lable2 = plt.scatter(X[pos,0],X[pos,1],marker='o',c='r')
    plt.legend((lable1,lable2),('Admitted','Not admitted'))
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()

def decisionBundary(theta):
    X,y = loadData()
    y = y.astype(int)
    neg = (y == 0)
    pos = (y == 1)
    lable1 = plt.scatter(X[neg,0],X[neg,1],marker='o',c='b')
    lable2 = plt.scatter(X[pos,0],X[pos,1],marker='o',c='r')
    plt.legend((lable1,lable2),('Admitted','Not admitted'))
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    boundary_xs = np.array([np.min(X[:,0]), np.max(X[:,0])])
    boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)
    plt.plot(boundary_xs,boundary_ys,'b-',label='Decision Boundary')
    plt.show()



if __name__ == "__main__":

    X,y = loadData()
    plotScatter()
    m,n = X.shape
    X = np.c_[np.ones((m,1)), X]  # add one col
    #y = np.reshape(y,(m,1))
    y= np.c_[y]
    initial_theta = np.zeros(X.shape[1])
    res = op.minimize(costFunction, initial_theta, args=(X,y), method='TNC', jac=gradient, options={'maxiter':400})
    theta = res.x
    print theta
    decisionBundary(theta)
