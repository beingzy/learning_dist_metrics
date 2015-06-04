"""
Collection of Distance Metrics
"""
import numpy as np
import scipy as sp
import pandas as pd
import itertools as it
from numpy import sqrt
from scipy.spatial.distance import euclidean



def weighted_euclidean(x, y, w=None):
    """return weigthed eculidean distance of (x, y) given weights (w)

    distance = sqrt( sum( w[i]*(x[i] - y[i])^2 ) )
    """
    if len(x) != len(y):
        print "ERROR: length(x) is different from length(y)!"

    if w is None:
        w = [1] * len(x)

    x = [sqrt(i_w) * i_x for i_w, i_x in zip(w, x)]
    y = [sqrt(i_w) * i_y for i_w, i_y in zip(w, y)]
    return euclidean(x, y)


def pairwise_dist_wrapper(pair, data, weights=None):
    """ Return distance of two points (rows in matrix-like data), provided
        with a pair of row numbers.

        Parameters:
        -----------
        * pair: {list, integer}, a vector of two integers, denoting the row
            numbers
        * data: {matrix-like}, matrix stores observations
        * weights: {vector-like}, a weights vector for weighting euclidean
            distance

        Returns:
        -------
        res: {float}, a weighted euclidean distancce between two data points

        Examples:
        --------
        p = (0, 10)
        w = [1] * data.shape[1]
        dist = pairwise_dist_wrapper(pair, data, w)
    """
    if isinstance(data, pd.DataFrame):
        data = data.as_matrix()

    if weights is None:
        weights = [1] * data.shape[1]

    a = data[pair[0], :]
    b = data[pair[1], :]
    return weighted_euclidean(a, b, weights)


def all_pairwise_dist(pair_list, data, weights=None):
    """ Return a series of weighted euclidean distances of the pairs of
        data points specified by pair_list

        Parameters:
        ----------
        * pair_list: {list, list}, a list of pairs of integers (row index)
        * data: {matrix-like}, matrix stores observations
        * weights: {vector-like}, a weights vector for weighting euclidean
            distance

        Returns:
        --------
        dist: {vector-like, float} a series of ditances

        Examples:
        --------
        p = [[0, 1], [0, 4], [3, 4]]
        dist = all_pairwise_dist(p, data)
    """
    dist = []
    for i in pair_list:
        dist.append(pairwise_dist_wrapper(i, data, weights))
    return dist


def sum_grouped_dist(pair_list, data, weights=None):
    """ Return the sum of distance

        Parameters:
        ----------
        * pair_list: {list, list}, a list of pairs of integers
        * data: {matrix-like}, matrix stores observations
        * weights: {vector-like}, a weights vector for weighting euclidean
            distance

        Returns:
        --------
        dist: {vector-like, float} a series of ditances

        Examples:
        --------
        p = [[0, 1], [0, 4], [3, 4]]
        sum_dist = sum_grouped_dist(p, data)
    """
    return sum(all_pairwise_dist(pair_list, data, weights))


def squared_sum_grouped_dist(pair_list, data, weights=None):
    """ Return the sum of squared distance

        Parameters:
        ----------
        * pair_list: {list, list}, a list of pairs of integers
        * data: {matrix-like}, matrix stores observations
        * weights: {vector-like}, a weights vector for weighting euclidean
            distance

        Returns:
        --------
        dist: {vector-like, float} a series of ditances

        Examples:
        --------
        p = [[0, 1], [0, 4], [3, 4]]
        sum_dist = squared_sum_grouped_dist(p, data)
    """
    dist = all_pairwise_dist(pair_list, data, weights)
    dist_squared = [d * d for d in dist]
    return sum(dist_squared)
