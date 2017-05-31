import numpy as np
import struct
import os
import idx2numpy
import matplotlib.pyplot as plt



def getMnistDataSimple(ddir):

    train_labels = idx2numpy.convert_from_file(ddir + "train-labels.idx1-ubyte")

    train_images = idx2numpy.convert_from_file(ddir + "train-images.idx3-ubyte")

    test_labels = idx2numpy.convert_from_file(ddir + "t10k-labels.idx1-ubyte")

    test_images = idx2numpy.convert_from_file(ddir + "t10k-images.idx3-ubyte")

    return train_labels, train_images, test_labels, test_images

ddir = '/home/aaronb/Codebrary/Python/pynist/data/raw/'
train_labels, train_images, test_labels, test_images = getMnistDataSimple(ddir)
#
# # Compare shapes of raw MNIST arrays with Octave example arrays:
# np.shape(train_labels)
# np.shape(train_images)
# np.shape(test_labels)
# np.shape(test_images)
#
# #It seems like the code should work fine for the raw structure as well.
# # Just add a '.flatten()' after reading in each image.
# print train_labels[100]
#
#
#
# X = train_images
# y = train_labels
# np.shape(y)
# y = np.reshape(y,(np.size(y),1))
# X_re = np.reshape(X,(60000,28**2)).copy()

def reshapeImages(X, width=28, n_imgs=60000):

    return np.reshape(X,(n_imgs,width**2)).copy()

    # Given image widths and quantity, reshape into 2d array:
X = reshapeImages(X)


def sigmoid(z):

    return 1 / (1 + np.exp(-z))

def sigmoidGrad(z):

    return sigmoid(z)*(1-sigmoid(z))

def binaryMapper(y, nlabels=10):

    m = np.size(y)

    # Map labels to binary label vectors
    y_temp    = np.arange(0,nlabels)
    y_broad   = np.ones((np.size(y),nlabels))

    return np.array(y*y_broad==y_temp, dtype=int)
# np.size(y)
# np.arange(0,10)
# binaryMapper(y)
# y
# y
# np.shape(y)

def forwardProp(X,theta1, theta2):

    a1 = np.insert(X,0,1, axis=1)

    a2 = np.insert(
        sigmoid(theta1.dot(a1.T)),
                    0,1, axis=0)

    a3 = sigmoid(theta2.dot(a2))

    return a1, a2, a3

#a1, a2, a3 = forwardProp(X, theta1, theta2)



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


# nneurons = 25
# nlabels = 10
# theta1 = np.zeros((nneurons,np.size(X[0])+1))
# theta2 = np.zeros((nlabels,nneurons+1))
#
# costFunctionNe(X, y,theta1, theta2, lam=1, reg=True)
#
#
#
#  np.shape(theta2)
#
# J, a1, a2, a3 = costFunctionNe(X,y,theta1, theta2, lam=1, reg=True)
# s3 = a3 - binaryMapper(y).T
# np.shape(s3)
# pr
# # np.shape(theta2)
# # np.shape(a2)
# # np.shape(a1)
# # np.shape(theta1)
# # np.shape(theta2.dot(a2))
# # np.shape(theta1.dot(a1.T))
# # # The second layer:
# # np.shape(theta2.T.dot(s3))
# # s2 = theta2.T.dot(s3)[1:]*sigmoidGrad(theta1.dot(a1.T))
# #
# # np.shape(theta2.T.dot(s3))
# #
# # d1 = s2.dot(a1)
# # np.shape(d1)
# # np.shape(theta1)
# # d2 = s3.dot(a2.T)
# # np.shape(d2)
# # np.shape(theta2)
#
# #*sigmoidGrad(theta1.dot(a1.T))


def backProp(a1, a2, a3, theta1, theta2, y, reg=True, lam=1):

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

        d1 += (lam/m)*theta1
        d2 += (lam/m)*theta2

    return d1, d2

# d1, d2 = backProp(a1, a2, a3, y)
#
# np.min(theta1)+np.max(theta1)
# np.median(theta1)
#
#
# plt.hist(theta1.flatten())
# plt.show()
#
# theta1_init = np.random.standard_normal(size=np.shape(theta1))
# theta1_init
#
#
# fig = plt.figure()
# plt.scatter(0,0)
# plt.scatter(1,1)
# plt.show()
# plt.close()
#
# Jplot, = plt.plot([],[])
#
# plt.xlim([0,100])
# plt.show()
#
# plt.show()
# np.max(y)
# np.size(X[0])

def costLowerer(X, y, nneurons=25, nlabels=10, alpha=0.001, num_iters=100, lam=1, reg=True):
    ## A simple minimization function:

    # theta1 and theta2 are just templates giving the parameter matrix dimensions

    # Initialize the parameters- use zeroes or random:
    ## Random initialization
    #theta1 = np.random.standard_normal(size=(nneurons,np.size(X[0])+1))/2
    theta1 = np.random.standard_normal(size=(nneurons,np.size(X[0])+1))/2
    theta2 = np.random.standard_normal(size=(nlabels,nneurons+1))/10

    ## Zeroes initialization
    #theta1 = np.zeros((nneurons,np.size(X[0])+1))
    #theta2 = np.zeros((nlabels,nneurons+1))

    # Intialize the cost plot:
    # Jplot, = plt.plot([],[])
    # plt.xlim([0,num_iters])
    # plt.show()

    for i in range(0,num_iters):

        # Forward pass (get the cost):
        J, a1, a2, a3 = costFunctionNe(X,y,theta1, theta2, lam=lam, reg=reg)

        # Reverse pass (get the gradients):
        grad1, grad2 = backProp(a1, a2, a3, theta1,theta2, y, lam=lam, reg=reg)

        # Take a learning-rate-sized step along the gradients:
        ## These updates need to be simultaneous!
        theta1_ = theta1 - grad1*alpha
        theta2_ = theta2 - grad2*alpha

        theta1 = theta1_
        theta2 = theta2_

        # Jplot.set_ydata(np.append(Jplot.get_ydata(),J))
        # Jplot.set_xdata(np.append(Jplot.get_xdata(),i))
        # plt.draw()


    return theta1, theta2, J, a1, a2, a3
# Test the minimizer:
theta1, theta2, J, a1, a2, a3 = costLowerer(X,y, nneurons = 100, alpha = 1e-5,  num_iters=200)
#
print J
# np.shape(a3)
#
# a3_max = np.max(a3, axis=0)
#
# a3_labels = np.where(a3_max == a3)[0]
#
# np.shape(a3_labels)
#
# print a3_labels
# np.shape(y)
#
#
#
# np.shape(y.flatten()==a3_labels)
# y.flatten()==a3_labels
# np.count_nonzero(y.flatten()==a3_labels)

def outputMapper(output, expected):

    # Maps the output layer back to the labels and gives the F1 score:
    m = np.size(y)
    # Take just the elements on each row with the highest value:
    ## In other words, the highest probability det. by the NN

    output_maxprob = np.max(output, axis=0)

    # Convert output into a simple list that gives the label for each example
    ## that corresponds to the highest probability output

    output_label = np.where(output_maxprob == output)[0] # 'where' gives the full coordinates. we just need the "x" component.

    # Now we measure the success of the model via the F1 score:
    ##

    result = expected.flatten()==output_label

    n_correct = np.count_nonzero(result)

    score = (float(n_correct)/m)*100


    print "Score: "+str(score)+"% correct labels"
    return output_label, result, score


output_label, result score = outputMapper(a3,y)
