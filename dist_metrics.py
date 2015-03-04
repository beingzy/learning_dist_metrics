"""
Collection of Distance Metrics
"""
import numpy as np
import scipy as sp
import pandas as pd
import itertools as it


def weighted_euclidean(x, y, w = None):
    """return weigthed eculidean distance of (x, y) given weights (w)
    
    distance = sqrt( sum( w[i]*(x[i] - y[i])^2 ) )
    """
    from numpy import sqrt
    from scipy.spatial.distance import euclidean

    if len(x) != len(y):
    	print "ERROR: length(x) is different from length(y)!"

    if w is None:
    	w = [1] * len(x)

    x = [sqrt(i_w) * i_x for i_w, i_x in zip(w, x)]
    y = [sqrt(i_w) * i_y for i_w, i_y in zip(w, y)]
    return euclidean(x, y)

def pairwise_dist_wrapper(pair, data, *arg):
	"""Return distance of two points by referring to data by 
	   index specified by pair
	"""
	if isinstance(data, pd.DataFrame):
		data = data.as_matrix()
	a = data[pair[0], :]
	b = data[pair[1], :]
	w = arg[0]
	return weighted_euclidean(a, b, w)

def all_pairwise_dist(pair_list, data, w):
	"""
	"""
	dist = []
	for i in pair_list:
		dist.append(pairwise_dist_wrapper(i, data, w))
	return dist

def sum_grouped_dist(pair_list, data, w):
	"""Return the sum of distance
	"""
	return sum(all_pairwise_dist(pair_list, data, w))

def squared_sum_grouped_dist(pair_list, data, w):
	"""Return the sum of squared distance
	"""
	dist = all_pairwise_dist(pair_list, data, w)
	dist_squared = [ i * i for i in dist]
	return sum(dist_squared)
