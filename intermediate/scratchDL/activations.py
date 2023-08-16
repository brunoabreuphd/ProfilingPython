###
# File: activations.py
# Description: 
# Author: Bruno R. de Abreu  |  babreu at illinois dot edu
# National Center for Supercomputing Applications (NCSA)
#  
# Creation Date: Monday, 31st October 2022, 2:23:46 pm
# Last Modified: Monday, 31st October 2022, 2:23:48 pm
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

def relu(x):
    """
    Computes the ReLU activation function.

    Arguments:
        - x: input array

    Returns:
        - relu activation value
    """
    return np.maximum(0, x)

def softmax(x):
    """
    Computes the softmax activation function.

    Arguments:
        - x: input array

    Returns:
        - softmax activation value
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def deriv_relu(x):
    """
    Computes the derivative of the ReLU activation function.

    Arguments:
        - x: input array

    Returns:
        - derivative of the relu activation value
    """
    return 1. * (x > 0)

def deriv_softmax(x):
    """
    Computes the derivative of the softmax activation function.

    Arguments:
        - x: input array

    Returns:
        - derivative of the softmax activation value
    """
    dx = np.exp(x) / sum(np.exp(x)) * (1 - np.exp(x) / sum(np.exp(x)))
    return dx