# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
import costs as cst

def compute_gradient(y, tx, w):
    """Computes the gradient at w for Gradient Descent.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # Error vector
    e = y - tx @ w
    # N = y.shape[0]
    return -1/y.shape[0] * np.transpose(tx) @ e

# I think adding the optional parameter does not break the function tester
def gradient_descent(y, tx, initial_w, max_iters, gamma, print = False): 
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Initalization of w for first iteration
    w = initial_w
    for n_iter in range(max_iters):
        # Compute gradient
        e = y - tx @ w
        g = -1/y.shape[0] * np.transpose(tx) @ e
        
        # update w
        w = w - gamma * g

        ######### Set flag true to show trace, this could be removed to avoid checking the if we wanta bit more speed
        if print:
            loss = cst.compute_loss_MSE(y,tx,w)
            print(
                "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
                )
            )

    return w, cst.compute_loss_MSE(y,tx,w)
