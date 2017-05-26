import numpy as np

def loadTestData():

    from scipy.io import matlab

    testPath    = '/home/aaronb/Downloads/Coursera/machinelearning/machine-learning-ex4/ex4/'
    dataFile    = 'ex4data1.mat'
    weightsFile = 'ex4weights.mat'

    data    = matlab.loadmat(testPath+dataFile)
    weights = matlab.loadmat(testPath+weightsFile)

    y = data['y']
    X = data['X']

    theta1 = weights['Theta1']
    theta2 = weights['Theta2']

    return X, y, theta1, theta2

X, y, theta1, theta2 = loadTestData()

def sigmoid(z):

    return 1 / (1 + np.exp(-z))

def sigmoidGrad(z):

    g_ = 0

    g_ = sigmoid(z)*(1-sigmoid(g))

    return g_



def costFunction(X, y,theta, lam=None, reg=False):
    m = len(y)

    J = 0
    grad = np.zeros(np.size(theta))

    h = sigmoid(X*theta)

    # without regularization:
    if reg==False:
        J = (1/m)*np.sum(-y*np.log(h)-(1-y)*log(1-h))
        grad = (1/m) * np.T(X)*(h-y)
        return J, grad

    # with regularization:
    elif reg==True:
            J = J + (lam/(2.0*m))*np.sum(theta[1:]**2)
            grad[1:] = grad[1:]+(lam/m)*(theta[1:])
            return J,grad

    else:
        print "reg should be set to True or False to turn regularization on or off"
        pass


def costFunctionNe(X, y,theta1, theta2, lam=None, reg=False):

    m = np.size(y)
    y_temp    = np.arange(1,11,1)
    y_broad   = np.ones([10,np.size(y)])
    y_boolmap = np.array(y*y_broad.T==y_temp, dtype=int)

    y = y_boolmap.copy()
    #y_boolmap.shape()


    a1 = np.insert(X,0,1, axis=1)

    a2 = np.insert(
        sigmoid(np.matmul(theta1,a1.T)),
                    0,1, axis=0)

    a3 = sigmoid(np.matmul(theta2,a2))

    h = a3


    # without regularization:
    if reg==False:
        J = np.sum(
            np.matmul(np.log(h),y)-np.matmul(np.log(1-h),(1-y))
                        )/m
        grad = np.matmul(X.T,(h.T-y))/m
        print J
        return J, grad, h, a2, a1, m, y


    # with regularization:
    elif reg==True:
            J = J + (lam/(2.0*m))*np.sum(theta[1:]**2)
            grad[1:] = grad[1:]+(lam/m)*(theta[1:])
            return J, grad, h, a2, a1, m, y

    else:
        print "reg should be set to True or False to turn regularization on or off"
        pass




J, grad, h, a2, a1, m, y_map = costFunctionNe(X,y,theta1, theta2)


y_map.shape
