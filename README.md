# pynist
A simple 3-layer neural network implementation in python. 
Inspired by Coursera Machine Learning course's excercises, which used Octave/matlab.)
I.e., the cost function used here and the basic theory follow the equations described in that course.

Running the code from within iPython is recommended (so that the figure windows stay open after the code finishes):

```
%run nnLabeler.py 400 28 10 0.00001 1 True
```
Will run the script for 400 iterations, with 28 hidden units, assuming 10 labels (digits), a learning rate of 1e-5, lambda of 1, and with regularization.

However you can also just run via `python nnLabeler.py 400 28 10 0.00001 1 True` from the command shell, and check the output .pdf plots afterwards.

Doing so, and waiting through all of the iterations, will produce terminal output looking like this:

```
Iteration #398 of 400
Regularized Cost: 0.508066490148
Iteration #399 of 400
Getting training set score..
Number of samples checked: 60000
Score: 92.8716666667% correct labels
Calculating test-set score...
Loading in testing data
Images succesfully loaded! (As a (10000, 784) numpy array)
Labels succesfully loaded! (As a (10000,) numpy array)
Number of samples checked: 10000
Score: 92.78% correct labels
```

After running the code, the results-dictionary can be recovered using Pickle, to check the actual output labels, etc.:

```
In [6]: with open('./history/result_[timestamp].p', 'rb') as f:
   ...:          pickle.load(f)
   ...:     
```

The contents of the pickle are in the form of a dictionary `results`, as follows:

```
    results =  {'theta1':theta1,
                'theta2':theta2,
                'output_label':output_label,
                'score': score,
                'outout_label_test': output_label_test,
                'result': result_test,
                'score_test' : score_test}
```                

Pickle files are saved after a successful run, with timestamp of the run. So just choose the '.p' file with the most recent timestamp, if you just ran the code.
The source code for each run is also copied (with the same timestamp) so that results are always accompanied by the code that generated them. 
Plots are stored in `./plots/*.pdf` also with a timestamp.

After downloading the raw MNIST training and test data, 'nnLabeler' allows for a simple batch gradient descent
optimization (with the cost given by 'costFunction' after forward feeding with 'forwardProp') of the neural network hidden layer weights, `theta1` and `theta2`,
along their respective gradients (determined by back-propagation, using 'backProp').

The input data is standardized to improve the optimization process, from its original range of 0-255.

After fitting the training data (60,000 images) and finding a training accuracy score, the model will
be applied to the test data to determine a cross-test score.

Using the code as written, applying regularzation with a lambda of 1.0, a learning rate of 1e-5, and for 400
iterations of batch gradient descent, you may find an accuracy of about ~92% for training and cross-testing:

The profile of cost vs. iteration is saved in `plots/J_progress_[timestamp].pdf`

The code will print the cost for each iteration, as well as live-plot the cost vs. iteration number. After training,
the weights (reconstructed form 'theta1' and 'theta2', and shaped into the original image dimensions) will be displayed (and saved as 
`plots/weights_[timestamp].pdf`
This will show how well the model is able to "understand" what the digits in the dataset look like. 

Also, histograms (`plots/resHist_[timestamp].pdf`) of both the modeled labels and the actual labels will be plotted. This is mainly just for diagnostic purposes
if bugs are encountered, or to use for future improvements of the code. You can quicky see from such a plot which labels
have large discrepancies.
