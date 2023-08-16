###
# File: propagations.py
# Description: 
# Author: Bruno R. de Abreu  |  babreu at illinois dot edu
# National Center for Supercomputing Applications (NCSA)
#  
# Creation Date: Monday, 31st October 2022, 2:29:30 pm
# Last Modified: Monday, 31st October 2022, 2:29:33 pm
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

from utils import one_hot

def forward_prop(x, params):
    """
    Forward propagation for L layers.
    All hidden layers are ReLU-activated except for the last one, which is softmax.
    Arguments:
        - x: input array
        - params: dictionary of parameters
    Returns:
        - A (dict): the activations
    """
    L = len(params) // 2

    activations = {}
    activations['A0'] = x

    for l in range(1,L):
        activations['Z'+str(l)] = np.dot(params['W'+str(l)], activations['A'+str(l-1)]) + params['b'+str(l)]
        activations['A'+str(l)] = relu(activations['Z'+str(l)]) 

    # now the last layer
    activations['Z'+str(L)] = np.dot(params['W'+str(l)], activations['A'+str(L-1)]) + params['b'+str(L)]
    activations['A'+str(L)] = softmax(activations['Z'+str(L)])

    return activations


def backward_prop(activations, params, y):
    """
    Backward propagation for L layers.
    Arguments:
        - activations: dictionary of activations
        - params: dictionary of parameters
        - y: labels
    Returns:
        - grads (dict): the gradients
    """
    L = len(params) // 2
    one_hot_y = one_hot(y)
    m = one_hot_y.shape[1]

    derivs = {}
    grads = {}

    # last layer
    derivs['dZ'+str(L)] = (activations['A'+str(L)] - one_hot_y)
    grads['dW'+str(L)] = (1./m) * np.dot(derivs['dZ'+str(L)], activations['A'+str(L-1)].T)
    grads['db'+str(L)] = (1./m) * np.sum(derivs['dZ'+str(L)])

    # all the other layers
    for l in reversed(range(1,L)):
        

    return