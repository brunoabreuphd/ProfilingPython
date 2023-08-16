###
# File: utils.py
# Description: 
# Author: Bruno R. de Abreu  |  babreu at illinois dot edu
# National Center for Supercomputing Applications (NCSA)
#  
# Creation Date: Monday, 31st October 2022, 2:17:16 pm
# Last Modified: Monday, 31st October 2022, 2:17:27 pm
#  
# Copyright (c) 2022, Bruno R. de Abreu, National Center for Supercomputing Applications.
# All rights reserved.
# License: This program and the accompanying materials are made available to any individual
#          under the citation condition that follows: On the event that the software is
#          used to generate data that is used implicitly or explicitly for research
#          purposes, proper acknowledgment must be provided in the citations section of
#          publications. This includes both the author's name and the National Center
#          for Supercomputing Applications. If you are uncertain about how to do
#          so, please check this page: https://github.com/babreu-ncsa/cite-me.
#          This software cannot be used for commercial purposes in any way whatsoever.
#          Omitting this license when redistributing the code is strongly disencouraged.
#          The software is provided without warranty of any kind. In no event shall the
#          author or copyright holders be liable for any kind of claim in connection to
#          the software and its usage.
###

import numpy as np

def initialize_parameters(layers_dims):
    """
    Initializes parameters randomly. Returns dictionary with parameters.

    Arguments:
        - layers_dims: list of integers, each integer represents the size of the layer
    
    Returns:
        - parameters: dictionary containing the parameters W and b for each layer
    """
    parameters = {}

    for layer in range(1, len(layers_dims)):
        parameters['W' + str(layer)] = np.random.randn(layers_dims[layer], layers_dims[layer-1]) * np.sqrt(1. / layers_dims[layer])
        parameters['b' + str(layer)] = np.random.randn(layers_dims[layer], 1) * np.sqrt(1. / layers_dims[layer])

    return parameters


def one_hot(y):
    """
    Converts a vector of labels into a one-hot matrix.

    Arguments:
        - y: vector of labels
    
    Returns:
        - one_hot: one-hot matrix
    """
    one_hot = np.zeros((y.shape[0], y.max() + 1))
    one_hot[np.arange(y.shape[0]), y] = 1

    return one_hot.T


def update_parameters(params, grads, alpha):
    """
    Updates the parameters of the network using gradient descent.

    Arguments:
        - params: dictionary containing the parameters W and b for each layer
        - grads: dictionary containing the gradients dW and db for each layer
        - alpha: learning rate

    Returns:
        - params_updated: dictionary containing the updated parameters W and b for each layer
    """

    # number of layers
    L = len(params) // 2

    params_updated = {}
    for l in range(1, L+1):
        params_updated['W' + str(l)] = params['W' + str(l)] - alpha * grads['dW' + str(l)]
        params_updated['b' + str(l)] = params['b' + str(l)] - alpha * grads['db' + str(l)]

    return params_updated


def cross_entropy(Y_one_hot, Y_hat, epsilon=1e-12):
    """
    Computes cross entropy between the target Y_one_hot (encoded) and the output Y_hat (predicted).

    Arguments:
        - Y_one_hot: one-hot matrix of the target
        - Y_hat: predicted output
        - epsilon: small value to avoid division by zero

    Returns:
        - crent: cross entropy
    """

    # clip to avoid division by zero/one
    Y_hat = np.clip(Y_hat, epsilon, 1. - epsilon)

    # compute cross entropy
    crent = np.sum(Y_one_hot * np.log(Y_hat), axis=0)
    crent = -np.mean(crent)

    return crent