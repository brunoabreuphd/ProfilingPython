###
# File: eigvals.py
# Description: Calculate eigenvalues of random matrices using Scipy
# Author: Bruno R. de Abreu  |  babreu at illinois dot edu
# National Center for Supercomputing Applications (NCSA)
#  
# Creation Date: Thursday, 12th May 2022, 12:55:42 pm
# Last Modified: Tuesday, 17th May 2022, 9:30:16 am
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

from scipy import linalg
import numpy as np

def create_matrix(matOrder):
    mat = np.random.rand(matOrder, matOrder)
    return mat

@profile
def calculate_eigvals(mat):
    eigvals = linalg.eigvals(mat)
    return eigvals

matOrder = 2**10
mat = create_matrix(matOrder)
eigvals = calculate_eigvals(mat)
print(eigvals[0:10])
