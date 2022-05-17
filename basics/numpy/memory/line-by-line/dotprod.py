###
# File: dotprod.py
# Description: Perform dot product between two random vectors
# Author: Bruno R. de Abreu  |  babreu at illinois dot edu
# National Center for Supercomputing Applications (NCSA)
#  
# Creation Date: Thursday, 12th May 2022, 9:46:42 am
# Last Modified: Tuesday, 17th May 2022, 9:16:44 am
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

@profile
def create_vectors(vecSize):
    v1 = np.random.rand(vecSize)
    v2 = np.random.rand(vecSize)
    return v1, v2

def calculate_dp(v1,v2):
    return np.dot(v1, v2)

vecSize = 2**27
v1, v2 = create_vectors(vecSize)
dp = calculate_dp(v1,v2)

print(f" Dot product is {dp}")
