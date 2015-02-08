"""
Learning Distance Metrics Algorithm

Author: Yi Zhang <beingzy@gmail.com>
Create Date: Feb/04/2015
"""
import time
import numpy as np 
import pandas as pd 
import scipy as sp 
from itertools import combinations
from scipy.optimize import minimize

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

    def __init__(self, dist_func = None, report_excution_time= True):
    	self._transform_matrix = np.array([])
    	self._ratio = 1
    	self.report_excution_time = report_excution_time
        pass 

    def fit(self, X, S, D = None):
        """Fit the model with X and given S and D

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
        self._fit(X, S, D)
        return self 

    def fit_transform(self, X, S, D = None):
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
    	self.fit(X, S, D)
    	X_new = self.transform(X)
    	return X_new

    def _fit(self, X, S, D = None):
        """Fit the model with given information: X, S, D


        Fit the learning distance metrics: (1) if only S is given,
        all pairs of items in X but not in S are considered as in D;
        (2) if both S and D given, items in X but neither in S nor in D
        will be removed from fitting process.

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
        n_sample, n_features = X.shape

        bnds = [(0, None)] * n_features # boundaries
        init = [1] * n_features # initial weights

        if D is None:
            all_pairs = [i for i in combinations(range(n_sample), 2)]
            D = get_exclusive_pairs(all_pairs, S)
        else:
            covered_items = get_unique_items(S, D)
            X = np.delete(X, covered_items, 0)

        def objective_func(w):
            a = squared_sum_grouped_dist(S, X, w) * 1.0
            b = squared_sum_grouped_dist(D, X, w) * 1.0
            return a / b

        start_time = time.time()
        fitted = minimize(objective_func, init, method="L-BFGS-B", bounds = bnds)
        duration = time.time() - start_time

        if self.report_excution_time:
            print("--- %s seconds ---" % duration)

        self._transform_matrix = fitted.x
        self._ratio = fitted.fun / objective_func(init)
        self._dist_func = lambda x, y: weighted_euclidean(x, y, w = w)

        return (self._transform_matrix, self._ratio)

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
    	n_sample, n_features = X.shape
    	trans_matrix = self._transform_matrix
    	if not len(trans_matrix) == n_features:
    		raise ValueError('Transformation matrix is not compatiable with X!')
    	X_new = self._transform_matrix * X 

    	return X_new

    def get_transform_matrix(self):
    	"""Returned the fitted transformation matrix (A)

    	Returns:
    	-------
    	_trans_vec: {matrix-like, np.array}, shape(n_features, n_features)
    	       A transformation matrix (A) 
    	"""
    	return self._transform_matrix

    def fitted_dist_func(self, x, y):
        """Returned the distance functions used in fitting model 

        Returns:
        --------
        func: {function} a function accept (x1, x2, *arg)
        """     
        if not self._transform_matrix is None:
            w = self._transform_matrix
            g = lambda x, y: weighted_euclidean(x = x, y = y, w = w)
        return g(x, y)

    def get_ratio(self):
        """The ratio of aggregate metrics of similiar points 
           over the couterparts of different points
        """
        return self._ratio



def get_exclusive_pairs(target_pairs, reference_pairs):
    """ Remove from target_paris the item (pairs) which
        has matches in reference_pairs.

        Parameters:
        -----------
        target_pairs: {list}, [(1, 2), (1, 3), ...]
        reference_pairs: {list}, [(2, 1), (10, 11)]

        Returns:
        -------

    """
    res = list(target_pairs)
    for i, ref_pair in enumerate(reference_pairs):
        for j, tgt_pair in enumerate(target_pairs):
            if set(ref_pair) == set(tgt_pair):
                res.pop(j)
    return res

def get_unique_items(x_pairs, y_pairs):
    """Return all item mentioned either by x_pairs
       or y_pairs.
    """
    x_pairs.extend(y_pairs)
    res = []
    for a, b in x_pairs:
        if not a in res:
            res.append(a)
        if not b in res:
            res.append(b)
    return res



