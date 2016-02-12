'''
Created on Jan 29, 2016

@author: dc
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_CIFAR10
from memory_profiler import profile

class L1Distance(object):
    '''
    Show how to compute L! distance for both train and test
    '''


    def __init__(self):
        '''
        data_utils returns X_test, X_train where X_test is the CIFAR 100 images of 10k images
        X_train is 50k images
        y_train is 10k labels in integer form we have to correlate the int to a list position in meta_batch
        y_test is what we are supposed to label  
        '''
        os.chdir('/var/lib/jenkins/jobs/PythonTest/workspace/assignment1')
        
    def train(self,x,y):
        '''
        it may not be necessary to call train because we set Xtrain and Ytrain during init
        '''
        self.Xtrain=x
        self.Ytrain=y
    @profile(precision=4)
    def predict(self,Xtest):
        '''
        we didnt build a model w/the training data. we just remembered it
        take the test data and find the label for the closest training image. we have a set of 50k training data imaages and 10k test images
        '''
        num_test = X_test.shape[0]
        Ypred = np.zeros(num_test, dtype = self.Ytrain.dtype)

        # loop over all test rows
        for i in xrange(num_test):
        # find the nearest training image to the i'th test image
        # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtrain - Xtest[i,:]), axis = 1)
            min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.Ytrain[min_index] # predict the label of the nearest example
            if(i % 5==0):
                print i
        return Ypred
        
a= L1Distance()
cifar10_dir = '/root/cs231n/assignment1/cs231n/datasets/cifar-10-batches-py/' 
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
print 'X_train shape:', X_train.shape
print 'y_train shape:', y_train.shape
print 'X_test shape:', X_test.shape
print 'y_test shape:', y_test.shape
a.train(X_train,y_train)

#why do we need shape[0]?      
Xtr_rows=X_train.reshape(X_train.shape[0],3*32*32)
Xte_rows=X_test.reshape(X_test.shape[0],3*32*32)
a.train(Xtr_rows,y_train)
yte_predict=a.predict(Xte_rows)
print 'accuracy: %f' % ( np.mean(yte_predict == y_test) )

