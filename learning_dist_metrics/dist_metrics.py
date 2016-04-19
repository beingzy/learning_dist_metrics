"""
Collection of Distance Metrics
"""
import pandas as pd
import numpy as np
from numpy import sqrt
from distance_metrics.GeneralDistanceWrapper import GeneralDistanceWrapper


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
            D += (xi - yi) * (xi - yi) * wi
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
    sum_squared_dist = 0
    for pair in pair_list:
        dist = pairwise_dist_wrapper(pair, data, weights)
        sum_squared_dist += dist * dist
    return sum_squared_dist


class WeightedDistanceTester(object):

    def __init__(self, user_ids, user_profiles, sim_edges, diff_edges):
        self._general_dist_wrapper = GeneralDistanceWrapper()
        self._general_dist_wrapper.fit(user_profiles)
        self._sim_dist_array = self._get_1d_squared_diff(sim_edges, user_ids, user_profiles)
        self._diff_dist_array = self._get_1d_squared_diff(diff_edges, user_ids, user_profiles)

    def _get_1d_squared_diff(self, edges, user_ids, user_profiles):
        n_obs, n_feat = len(edges), user_profiles.shape[1]
        dist_1ds = np.empty((n_obs, n_feat))
        is_warned = False

        if isinstance(user_profiles, np.ndarray):
            for ii, (a_idx, b_idx) in enumerate(edges):
                try:
                    #a_idx = [i for i, uid in enumerate(user_ids) if uid == e[0]]
                    #b_idx = [i for i, uid in enumerate(user_ids) if uid == e[1]]
                    a_profile = user_profiles[a_idx, :]
                    b_profile = user_profiles[b_idx, :]
                    dist_1ds[ii, :] = self._general_dist_wrapper.get_difference(a_profile, b_profile)
                except:
                    dist_1ds[ii, :] = np.nan
                    is_warned = True

        elif isinstance(user_profiles, pd.DataFrame):
            for ii, (a_idx, b_idx) in enumerate(edges):
                try:
                    #a_idx = [i for i, uid in enumerate(user_ids) if uid == e[0]]
                    #b_idx = [i for i, uid in enumerate(user_ids) if uid == e[1]]
                    a_profile = user_profiles[a_idx, :]
                    b_profile = user_profiles[b_idx, :]
                    dist_1ds[ii, :] = self._general_dist_wrapper(a_profile, b_profile)
                except:
                    dist_1ds[ii, :] = np.nan
                    is_warned = True
        else:
            msg = " ".join(["user_profiles is in unsupported data structures",
                            "(it must be numpy.ndarray or pandas.DataFrame)!"])
            raise ValueError(msg)

        if is_warned:
            bool_idx = np.isnan(dist_1ds[:, 0]).tolist()
            idx = [i for i, val in enumerate(bool_idx) if val == False]
            dist_1ds = dist_1ds[idx, :]

        return dist_1ds

    def _get_sim_agg_info(self, weights):
        sim_sum_squared = (np.square(self._sim_dist_array * weights)).sum(axis=1).sum()
        return float(sim_sum_squared)

    def _get_diff_agg_info(self, weights):
        """ return sum of distance """
        diff_sum = np.sqrt((np.square(self._diff_dist_array * weights)).sum(axis=1)).sum()
        return float(diff_sum)

    def update(self, weights):
        sim_info = self._get_sim_agg_info(weights)
        diff_info = self._get_diff_agg_info(weights)
        return sim_info - diff_info