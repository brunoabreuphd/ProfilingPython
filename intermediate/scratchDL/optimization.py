###
# File: optimization.py
# Description: 
# Author: Bruno R. de Abreu  |  babreu at illinois dot edu
# National Center for Supercomputing Applications (NCSA)
#  
# Creation Date: Tuesday, 3rd January 2023, 2:41:10 pm
# Last Modified: Tuesday, 3rd January 2023, 2:41:14 pm
#  
# Copyright (c) 2023, Bruno R. de Abreu, National Center for Supercomputing Applications.
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

import utils
import metrics
import propagations as prop

def gradient_descent_optimization(X, Y, layers_size, max_iter, alpha):
    """
    Trains the neural network using gradient descent.

    Arguments:
        - X: input data
        - Y: labels
        - layers_size (list): a list containing the dimensions of each layer
        - max_iter (int): the maximum number of iterations
        - alpha (float): the learning rate
    """
    params = utils.initialize_parameters(layers_size)
    L = len(params) // 2
    
    accuracies = []
    losses = []

    for it in range(1, max_iter + 1):
        # compute activations
        activations = prop.forward_prop(X, params)
        # make predictions
        Y_hat = utils.get_predictions(activations["A" + str(L)])
        # compute accuracy
        accuracy = metrics.get_accuracy(Y_hat, Y)
        accuracies.append(accuracy)

        # compute loss
        loss = metrics.cross_entropy(utils.one_hot(Y), activations["A" + str(L)])
        losses.append(loss)

        # compute gradients
        grads = prop.backward_prop(activations, params, Y)

        # update parameters
        params = utils.update_parameters(params, grads, alpha)

        if it % 10 == 0:
            print('Accuracy at iteration {}: {}'.format(it, accuracy))

    
    return params, accuracies, losses
