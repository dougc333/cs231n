import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  D = W.shape[0]
  C = W.shape[1]
  N = X.shape[0]

  for i in range(N):
     cur_X = X[i,:] # we select X_i, which is a 1xD matrix
     f_scores = cur_X.dot(W)
     f_max = np.max(f_scores)
     # now we want to subtract everything by the f_max so we get a smaller f_scores with the same proportion
     f_scores -= f_max
      
     # Second, let's take care of the cost function, where it's equal to log(e^(score_i)/sum(e^(score_all)))
     loss += -f_scores[y[i]] + np.log(np.sum(np.exp(f_scores)))
      
     # Thirdly, let's take care of the cost function prime for W_i
     # it's equal to X_i(-1 + e^(score_i)/(sum(e^(score_all))))
     # important: I was confused about this but after thinking about it:
     # The COLUMN of W are the classes. You assign these d/dW_j for j = class # 
     # and to compute them you get a [1 x D] matrix from 
     # X_i * (scalar of np.exp(f_scores[j])/(np.sum(np.exp(f_scores)))
     for j in range(C):
       dW[:,j] += cur_X*(-1*(j==y[i]) + np.exp(f_scores[j])/(np.sum(np.exp(f_scores))))

  loss /= N # we need to average out the sample
  dW /= N
  loss += reg*np.sum(W**2)/2  # we need to add the regularization terms
  dW += reg*W
  


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

