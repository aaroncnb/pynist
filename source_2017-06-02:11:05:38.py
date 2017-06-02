import numpy as np
import struct
import os
import idx2numpy
import matplotlib.pyplot as plt
from shutil import copyfile
from datetime import datetime
import pickle

#import cudamat as cm


def getMnistTrain(ddir):

    train_labels = idx2numpy.convert_from_file(ddir + "train-labels.idx1-ubyte")

    train_images = idx2numpy.convert_from_file(ddir + "train-images.idx3-ubyte")

    # test_labels = idx2numpy.convert_from_file(ddir + "t10k-labels.idx1-ubyte")
    #
    # test_images = idx2numpy.convert_from_file(ddir + "t10k-images.idx3-ubyte")
    train_labels = np.reshape(train_labels,(np.size(train_labels),1))

    train_images = reshapeImages(train_images)
    return  train_images, train_labels


def reshapeImages(X, width=28, n_imgs=60000):
    # Given image widths and quantity, reshape into 2d array:

    return np.reshape(X,(n_imgs,width**2)).copy()

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

def forwardProp(X,theta1, theta2):

    a1 = np.insert(X,0,1, axis=1)

    a2 = np.insert(
        sigmoid(theta1.dot(a1.T)),
                    0,1, axis=0)

    a3 = sigmoid(theta2.dot(a2))

    return a1, a2, a3

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

        print "Regularized Cost: "+str(J)
    else:
        print "Unregularized Cost: "+str(J)

    return J, a1, a2, a3


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


def costLowerer(ddir, nneurons=100, nlabels=10, alpha=0.001, num_iters=10, lam=1, reg=True, rdm_init=True):

    # Get the starting time for labeling output files:
    time = datetime.now().strftime('%Y-%m-%d:%H:%M:%S')

    X, y = getMnistTrain(ddir)

    if rdm_init == True:
        ## Randomly initialize theta for symmetry-breaking:
        theta1 = np.random.standard_normal(size=(nneurons,np.size(X[0])+1))/10
        theta2 = np.random.standard_normal(size=(nlabels,nneurons+1))/10
    else:
        ## Zeroes initialization
        theta1 = np.zeros((nneurons,np.size(X[0])+1))
        theta2 = np.zeros((nlabels,nneurons+1))

    # Intialize the cost plot for visualizing convergence/non-convergence:
    #Jplot, = plt.plot([],[])
    plt.axis()
    plt.ion()
    plt.xlim([0,num_iters])

    Jplot = []
    iplot = []

    for i in range(0,num_iters):

        # Forward pass (apply activations and get the cost):
        J, a1, a2, a3 = costFunctionNe(X,y,theta1, theta2, lam=lam, reg=reg)

        # Reverse pass (get the gradients):
        grad1, grad2 = backProp(a1, a2, a3, theta1,theta2, y, lam=lam, reg=reg)

        # Take a learning-rate-sized step along the gradients:
        ## These updates need to be simultaneous!
        ###FUTURE WORK: Add an SGD option.
        theta1_ = theta1 - grad1*alpha
        theta2_ = theta2 - grad2*alpha

        theta1 = theta1_
        theta2 = theta2_

        print "Iteration #"+str(i)+" of "+str(num_iters)

        Jplot.append(J)
        iplot.append(i)

        plt.scatter(i,J)
        plt.pause(0.05)

    plt.savefig(ddir+"J_progress_"+time+".pdf")

    # Show test the results against the "true" labels:
    output_label, result, score = outputMapper(a3,y)

    # Show the final weights-images:
    showWeighImgs(ddir, time, theta1,theta2)

    # Save the parameter matrices:
    with open('result_'+time+'.pickle', 'w') as f:
        pickle.dump([theta1, theta2, J, a1, a2, a3, output_label, result, score], f)

    # Copy the source code for the current run and:
    copyfile('costFunction.py', 'source_'+time+'.py')

    return theta1, theta2, J, a1, a2, a3, output_label, result, score, X, y, output


def showWeighImgs(ddir, time, theta1, theta2):

    nlabels = np.size(theta2[:,0])

    filters = theta2[:,1:].dot(theta1[:,1:])

    fig = plt.figure(figsize=(11.69,8.27))

    for lbl in  range(0, nlabels):

        ax = fig.add_subplot(2,5,lbl+1)

        ax.imshow(filters.reshape(10,28,28)[lbl])

        aspect = abs(ax.get_xlim()[1] - ax.get_xlim()[0]) / abs(ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.set_aspect(aspect)

    fig.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.05)

    fig.savefig(ddir+"weights_"+time+".pdf")
    plt.show()

def outputMapper(output, expected):

    # Maps the output layer back to the labels and gives the F1 score:
    m = np.size(expected)
    print "Number of samples checked: "+str(m)
    # Take just the elements on each row with the highest value:
    ## In other words, the highest probability det. by the NN

    output_maxprob = np.max(output, axis=0)

    # Convert output into a simple list that gives the label for each example
    ## that corresponds to the highest probability output

    output_label = np.where(output_maxprob == output)[0] # 'where' gives the full coordinates. we just need the "x" component.


    result = expected.flatten()==output_label[0]

    n_correct = np.count_nonzero(result)

    score = (float(n_correct)/m)*100

    print "Test: Confirm output label range = "+str(np.min(output_label))+" "+str(np.max(output_label))
    print "Test: Confirm expected label range = "+str(np.min(expected))+" "+str(np.max(expected))
    print "Score: "+str(score)+"% correct labels"
    return output_label, result, score


def main():
        #ddir = '/work1/users/aaronb/Codebrary/Python/Projects/pynist/data/raw/'
        #ddir = '/home/aaronb/Codebrary/Pytexion/Projects/pynist/data/raw/'
        ddir = '/home/aaronb/Projectbrary/pynist/data/raw/'
        theta1, theta2, J, a1, a2, a3, output_label, result, score = costLowerer(ddir, nneurons = 50, lam=1, alpha = 1e-5, num_iters=10, reg=False, rdm_init=True)

        return theta1, theta2, J, a1, a2, a3, output_label, result, score, X, y, output

# if __name__ == '__main__':
#     main()
