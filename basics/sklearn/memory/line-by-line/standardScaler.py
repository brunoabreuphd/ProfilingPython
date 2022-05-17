###
# File: standardScaler.py
# Description: Standarize features using Sklearn
# Author: Bruno R. de Abreu  |  babreu at illinois dot edu
# National Center for Supercomputing Applications (NCSA)
#  
# Creation Date: Thursday, 12th May 2022, 1:16:26 pm
# Last Modified: Tuesday, 17th May 2022, 10:17:37 am
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
from sklearn.preprocessing import StandardScaler

def create_data(nrows, ncols):
    data = np.random.rand(nrows, ncols)
    return data

@profile
def standardize_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    scaled = scaler.transform(data)
    return scaled

nrows = 2**20
ncols = 2**6

data = create_data(nrows, ncols)
scaledData = standardize_data(data)

print(data[0:6, 0:6])
print(scaledData[0:6, 0:6])

