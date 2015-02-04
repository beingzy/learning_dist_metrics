"""
Python Implementation of Newton-Raphson method

Author: Yi Zhang
Date: Feb/03/2015
"""
import numpy as np 
import pandas as pd 
import scipy as sp 

from 

# Parameters:
# -----------
# data: {matrix-like}
# S: {vector-like, list}
# D: {vector-like, list}
# C: ?
data = np.ones((4, 4))

n_sample, n_feature = data.shape 
 
a = np.ones((n_feature, ))
X = data

# hyper-parameter
fudge = 0.000001
threshold1 = 0.001
reduction = 2

s_sum = # sum of squared d_ij
d_sum = # sum of squared d_ij

tt = 1 # ?
error = 1 # ?


