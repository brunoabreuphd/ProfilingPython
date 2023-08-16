###
# File: train.py
# Description: 
# Author: Bruno R. de Abreu  |  babreu at illinois dot edu
# National Center for Supercomputing Applications (NCSA)
#  
# Creation Date: Tuesday, 3rd January 2023, 3:02:13 pm
# Last Modified: Tuesday, 3rd January 2023, 3:02:15 pm
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

import data_loaders as dl
import optimization as opt
import matploblib.pyplot as plt

if __name__== "__main__":
    # load mnist data
    X_train, Y_train, X_test, Y_test = dl.load_mnist_data()

    # set network dimensions
    layers_dims = [784, 256, 128, 64, 10]
    max_iter = 500
    alpha = 0.1

    # train network
    params, acc, loss = opt.gradient_descent_optimization(X_train, y_train, layers_dims, max_iter, alpha)