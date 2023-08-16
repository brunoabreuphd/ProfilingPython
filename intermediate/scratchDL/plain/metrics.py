###
# File: metrics.py
# Description: 
# Author: Bruno R. de Abreu  |  babreu at illinois dot edu
# National Center for Supercomputing Applications (NCSA)
#  
# Creation Date: Tuesday, 3rd January 2023, 2:37:43 pm
# Last Modified: Tuesday, 3rd January 2023, 2:37:45 pm
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

import numpy as np

class Metrics:
    def __init__(self):
        pass

    def accuracy(self, Y_hat, Y):
        """
        Given the predicted classes Y_hat and the true classes Y, computes the accuracy.

        Arguments:
            - Y_hat (1,m)-array: predicted classes
            - Y (1,m)-array: true classes

        Returns:
            - accuracy: the accuracy of the prediction
        """
        return np.sum(Y_hat == Y) / Y.size
    

    def cross_entropy(self, Y_one_hot, Y_hat, epsilon=1e-12):
        """
        Computes cross entropy between the target Y_one_hot (encoded) and the output Y_hat (predicted probabilities).

        Arguments:
            - Y_one_hot: one-hot matrix of the target
            - Y_hat: predicted output probabilities
            - epsilon: small value to avoid division by zero

        Returns:
            - crent: cross entropy
        """

        # clip to avoid division by zero/one
        Y_hat = np.clip(Y_hat, epsilon, 1. - epsilon)
        
        # compute cross entropy
        crent = np.log(Y_hat)
        crent = Y_one_hot * crent
        crent = -np.sum(crent, axis=0)
        crent = np.mean(crent)

        return crent

