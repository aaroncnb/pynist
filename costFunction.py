import numpy as np

class PyNist(object):
    """A simple feed-forward NN mnist application"""
    def __init__(self, arg):
        super(, self).__init__()
        self.arg = arg

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



    def costFunction(X, y,theta, lam=None, reg=False):
        m = len(y)

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
            sigmoid(theta1.dot(a1.T)),
                        0,1, axis=0)

        a3 = sigmoid(theta2.dot(a2))

        h = a3


        # without regularization:
        if reg==False:
            J = np.sum(
                -np.log(h).dot(y)-np.log(1-h).dot(1-y)
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

np.log(h).dot(y)

J, grad, h, a2, a1, m, y_map = costFunctionNe(X,y,theta1, theta2)


y_map.shape
