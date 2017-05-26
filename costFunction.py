import numpy as np


def loadTestData():

    from scipy.io import matlab

    testPath    = '/home/aaronb/Downloads/Coursera/machinelearning/machine-learning-ex4/ex4/'
    dataFile    = testPath + 'ex4data1.mat'
    weightsFile = testPath + 'ex4weights.mat'

    data    = matlab.loadmat(dataFile)
    weights = matlab.loadmat(weightsFile)

    y = data['y']
    X = data['X']

    theta1 = weights['Theta1']
    theta2 = weights['Theta2']

    return X, y, theta1, theta2

X, y, theta1, theta2 = loadTestData()

def sigmoid(z):

    return 1 / (1 + np.exp(-z))

def sigmoidGrad(z):

    return sigmoid(z)*(1-sigmoid(g))

def binaryMapper(y):

    m = np.size(y)

    # Map labels to binary label vectors
    y_temp    = np.arange(1,11,1)
    y_broad   = np.ones([10,np.size(y)])

    return np.array(y*y_broad.T==y_temp, dtype=int)


def forwardProp(X,theta1, theta2):



    a1 = np.insert(X,0,1, axis=1)

    a2 = np.insert(
        sigmoid(theta1.dot(a1.T)),
                    0,1, axis=0)

    return sigmoid(theta2.dot(a2))

def costFunctionNe(X, y,theta1, theta2, lam=None, reg=False):
    # Get the number of training examples:
    m = np.size(y)
    # Map labels to binary vectors:
    y = binaryMapper(y).T
    # Feed it forward:
    h = forwardProp(X, theta1, theta2)

    # Get the cost without regularization:
    J = np.sum(
        -(np.log(h)*y)-(np.log(1-h)*(1-y))
                                    ) /m

    # with regularization:
    if reg==True:

        J = J + ( ( np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2) )*(lam/(2.0*m) ) )
        print "Regularized"
    else:
        print "Unregularized"

    print "Cost: "+str(J)
    return J

J = costFunctionNe(X,y,theta1, theta2)

J = costFunctionNe(X,y,theta1, theta2, lam=1, reg=True)


np.log(np.e)
y_map.shape


ymapped = binaryMapper(y)

ymapped
h

h*ymapped.T
