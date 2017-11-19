import numpy as np
from random import shuffle
from past.builtins import xrange

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
  train_num = X.shape[0]
  class_num = W.shape[1]
  # f is [1, C], dW is (D, C)
  for i in range(train_num):
      f = X[i].dot(W)
      f -= np.max(f)
      exp_f = np.exp(f)
      f_exp_sum = np.sum(exp_f)
      loss -= np.log(exp_f[y[i]] / f_exp_sum)
      for j in range(class_num):
        if j == y[i]:
          dW[:, j] += -X[i] + X[i] * exp_f[y[i]] / f_exp_sum
        else:
          dW[:, j] += X[i] * exp_f[j] / f_exp_sum

  loss /= train_num
  loss += reg * np.sum(W * W)
  dW /= train_num
  dW += reg * W * 2
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
  # X is [N, D]
  # W is [D, C]
  # f is [N, C]
  train_num = X.shape[0]
  class_num = W.shape[1]
  f = X.dot(W)
  f -= np.max(f, axis=1).reshape((-1, 1))
  exp_f = np.exp(f)
  f_exp_sum = np.sum(exp_f, axis=1).reshape((-1, 1))
  mask = range(train_num), y

  loss = - np.sum(np.log(exp_f[mask] / f_exp_sum)) 
  loss /= train_num
  loss += reg * np.sum(W * W)

  coeff = exp_f / f_exp_sum
  coeff[mask] -= 1
  dW = X.T.dot(coeff)
  dW /= train_num
  dW += reg * W * 2
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

