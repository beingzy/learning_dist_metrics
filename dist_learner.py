"""
Learning Distance Metrics Algorithm
"""

import numpy as np 
import pandas as pd 
import scipy as sp 

from dist_metrics import weighted_euclidean
from dist_metrics import pairwise_dist_wrapper
from dist_metrics import all_pairwise_dist
from dist_metrics import sum_grouped_dist
from dist_metrics import squared_sum_grouped_dist

class LDM(object):
    """Learning Distance Metrics (LDM)

    An aglorithm learning distance metrics in a supervsied
    context to tighten spreads of points known similar and
    amplify separation of points considered different in
    a transformed space. 

    Parameters:
    -----------

    Attributes:
    -----------
    _trans_vec: array, [n_features, ]
        The A matrix transforming the original datasets

    _ratio: float
        The ratio of sum distances of transformed points known similar
        over its couterpart of tarnsformed points known different
	"""

    def __init__(self, dist_func = None):
    	self._dist_func = dist_func
    	self._ratio = 0
        pass 

    def fit(self, X, S, D):
        """Fit the model with X and given S and D

        Parameters:
        -----------

        Retruns:
        --------
        """
        self._fit(X, S, D)
        return self 

    def fit_transform(self, X):
    	"""Fit the model with X, S, D and conduct transformation on X

    	Parameters:
    	-----------
    	X: {matrix-like, np.array}, shape (n_sample, n_features)
    		Training data, where n_samples is the number of n_samples
    		and n_features is the number of features 

    	Returns:
    	--------
    	X_new: {marix-like, np.array}, shape (n_sample, n_features)
    		The return of X transformed by fitted matrix A
    	"""
    	pass 

    def _fit(self, X, S, D):
        """Fit the model with given information: X, S, D

        Parameters:
        ----------
        X: {matrix-like, np.array}, shape (n_sample, n_features)
    	    Training data, where n_samples is the number of n_samples
    	    and n_features is the number of features 
        S: {vector-like, list} a list of tuples which define a pair of
                  data points known as similiar 
        D: {vector-like, list} a list of tuples which define a pair of
                  data points known as different

        Returns:
        --------
        _trans_vec: {matrix-like, np.array}, shape(n_features, n_features)
    	       A transformation matrix (A) 
    	_ratio: float
        """
        pass 

    def transform(self, X):
    	"""Tranform X by the learned tranformation matrix (A)

    	Parameters:
    	-----------
    	X: {matrix-like, np.array}, shape (n_sample, n_features)
    		Training data, where n_samples is the number of n_samples
    		and n_features is the number of features 

    	Returns:
    	--------
    	X_new: {marix-like, np.array}, shape (n_sample, n_features)
    		The return of X transformed by fitted matrix A
    	"""
    	pass

    def get_transformation_matrix(self):
    	"""Returned the fitted transformation matrix (A)

    	Returns:
    	-------
    	_trans_vec: {matrix-like, np.array}, shape(n_features, n_features)
    	       A transformation matrix (A) 
    	"""

    def get_dist_func(self):
        """Returned the distance functions used in fitting model 

        Returns:
        --------
        func: {function} a function accept (x1, x2, *arg)
        """
        pass 

    def get_ratio(self):
        """The ratio of aggregate metrics of similiar points 
           over the couterparts of different points
        """
        pass self._ratio



