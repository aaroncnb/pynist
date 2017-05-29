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

    return sigmoid(z)*(1-sigmoid(z))

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

    a3 = sigmoid(theta2.dot(a2))

    return a1, a2, a3

a1, a2, a3 = forwardProp(X, theta1, theta2)

def costFunctionNe(X, y,theta1, theta2, lam=None, reg=False):
    # Get the number of training examples:
    m = np.size(y)
    # Map labels to binary vectors:
    y = binaryMapper(y).T
    # Feed it forward:
    a1, a2, a3 = forwardProp(X, theta1, theta2)

    # Get the cost without regularization:
    J = np.sum(
        -(np.log(a3)*y)-(np.log(1-a3)*(1-y))
                                    ) /m

    # with regularization:
    if reg==True:

        J += ( ( np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2) )*(lam/(2.0*m) ) )
        print "Regularized"
    else:
        print "Unregularized"

    print "Cost: "+str(J)
    return J, a1, a2, a3

J = costFunctionNe(X,y,theta1, theta2)

J = costFunctionNe(X,y,theta1, theta2, lam=1, reg=True)

s3 = a3 - binaryMapper(y).T
np.shape(s3)
np.shape(theta2)
np.shape(a2)
np.shape(a1)
np.shape(theta1)
np.shape(theta2.dot(a2))
np.shape(theta1.dot(a1.T))
# The second layer:
np.shape(theta2.T.dot(s3))
s2 = theta2.T.dot(s3)[1:]*sigmoidGrad(theta1.dot(a1.T))

np.shape(theta2.T.dot(s3))

d1 = s2.dot(a1)
np.shape(d1)
np.shape(theta1)
d2 = s3.dot(a2.T)
np.shape(d2)
np.shape(theta2)

#*sigmoidGrad(theta1.dot(a1.T))

def backProp(a1, a2, a3, y, reg=False, lam=None):

    # Get the 'error' for the third layer (aka first step of back-propagation):

    # Number of training examples
    m  = np.size(y)

    # The difference between expected and output values:
    ### (Input labels are mapped to binary vectors)
    s3 = a3 - binaryMapper(y).T

    # The second backprop layer: Applying the activation function's derivative:
    s2 = theta2.T.dot(s3)[1:]*sigmoidGrad(theta1.dot(a1.T))

    # Gradients along Theta 1 (should be the same dim as theta1!):
    d1 = s2.dot(a1)

    # Gradients along Theta 2 (should be the same dim as theta1!):
    d2 = s3.dot(a2.T)

    # Apply regularization if needed:
    ## Essentially just scale by the ratio of lambda to n-examples.
    if reg==True:

        d1 += (lam/m)*d1
        d2 += (lam/m)*d2

    return d1, d2

d1, d2 = backProp(a1, a2, a3, y)

np.min(theta1)+np.max(theta1)
np.median(theta1)
import matplotlib.pyplot as plt

plt.hist(theta1.flatten())
plt.show()

theta1_init = np.random.standard_normal(size=np.shape(theta1))
theta1_init


fig = plt.figure()
plt.scatter(0,0)
plt.scatter(1,1)
plt.show()
plt.close()

Jplot, = plt.plot([],[])
plt.xlim([0,100])
plt.show()

plt.show()

def costLowerer(X, y, theta1, theta2, alpha=0.0003, num_iters=100):
    ## A simple minimization function:

    # theta1 and theta2 are just templates giving the parameter matrix dimensions

    # Initialize the parameters- use zeroes or random:
    ## Random initialization
    #theta1 = np.random.standard_normal(size=np.shape(theta1))
    #theta2 = np.random.standard_normal(size=np.shape(theta2))

    ## Zeroes initialization
    theta1 = np.zeros(np.shape(theta1))
    theta2 = np.zeros(np.shape(theta2))

    # Intialize the cost plot:
    Jplot, = plt.plot([],[])
    plt.xlim([0,num_iters])
    plt.show()

    for i in range(0,num_iters):

        # Forward pass (get the cost):
        J, a1, a2, a3 = costFunctionNe(X,y,theta1, theta2, lam=0.3, reg=True)

        # Reverse pass (get the gradients):
        grad1, grad2 = backProp(a1, a2, a3, y)

        # Take a learning-rate-sized step along the gradients:
        ## These updates need to be simultaneous!
        theta1_ = theta1 - grad1*alpha
        theta2_ = theta2 - grad2*alpha

        theta1 = theta1_
        theta2 = theta2_

        Jplot.set_ydata(np.append(Jplot.get_ydata(),J))
        Jplot.set_xdata(np.append(Jplot.get_xdata(),i))
        plt.draw()


    return theta1, theta2

theta1, theta2 = costLowerer(X,y, theta1, theta2)
