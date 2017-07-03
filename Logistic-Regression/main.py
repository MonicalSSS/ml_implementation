from logistic_regression import *

def loadData():
    file1 = open("ex2data1.txt", "r")
    file2 = open("ex2data2.txt", "r")
    data = np.loadtxt(file1,dtype=float,delimiter=',', unpack=True)
    X = np.array(data[0:2,:])
    y = np.array(data[2,:])
    return X,y

def plotScatter(X, y):
    neg = (y == 0.0)
    pos = (y != 0.0)
    lable1 = plt.scatter(X[0,neg],X[1,neg],marker='o',c='b')
    lable2 = plt.scatter(X[0,pos],X[1,pos],marker='o',c='r')
    plt.legend((lable1,lable2),('Admitted','Not admitted'))
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()


if __name__ == "__main__":

    X,y = loadData()
    # plotScatter(X,y)
    n = len(X)
    m = len(X[0])  
    X = np.c_[np.ones(m), X.T] # add one col
    y = np.reshape(y,(1,m))

    initial_theta = np.zeros((n+1,1))
    cost,grad = costFunction(X,y,initial_theta)

    
