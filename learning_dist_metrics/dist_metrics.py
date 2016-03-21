"""
Collection of Distance Metrics
"""
import pandas as pd
from numpy import sqrt
from scipy.spatial.distance import euclidean
from numba import double
from numba import jit, autojit


def weighted_euclidean(x, y, w=None):
    """ return weigthed eculidean distance of (x, y) given weights (w)

    distance = sqrt( sum( w[i]*(x[i] - y[i])^2 ) )
    """
    if w is None:
        D = 0
        for xi, yi in zip(x, y):
            D += (xi - yi) * (xi - yi)
    else:
        D = 0
        for xi, yi, wi in zip(x, y, w):
            D += (xi - yi) * (xi - yi) * wi * wi
    return sqrt(D)


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


@autojit
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
    sum_dist = 0
    for pair in pair_list:
        sum_dist += pairwise_dist_wrapper(pair, data, weights)
    return sum_dist


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
