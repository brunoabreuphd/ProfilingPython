###
# File: data_loaders.py
# Description: 
# Author: Bruno R. de Abreu  |  babreu at illinois dot edu
# National Center for Supercomputing Applications (NCSA)
#  
# Creation Date: Tuesday, 3rd January 2023, 3:00:37 pm
# Last Modified: Tuesday, 3rd January 2023, 3:00:40 pm
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

import tensorflow as tf

class MNISTDataLoader():
    def __init__(self, one_hot_encode=False):
        self.one_hot_encode = one_hot_encode

    def load_train_data(self):
        """
        Loads the MNIST dataset.

        Returns:
            - X_train: training set
            - Y_train: training labels
            - X_test: test set
            - Y_test: test labels
        """
        # get it as it comes from keras
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

        # preprocess
        x_train = X_train.reshape(-1, 784).astype("float32") / 255.
        x_test = X_test.reshape(-1,784).astype("float32") / 255.
        y_train = Y_train.astype("int")
        y_test = Y_test.astype("int")


        return x_train.T, y_train.T