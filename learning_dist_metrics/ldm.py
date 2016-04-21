"""
Learning Distance Metrics Algorithm

Author: Yi Zhang <beingzy@gmail.com>
Create Date: Feb/04/2015
"""
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.random import choice

from learning_dist_metrics.dist_metrics import weighted_euclidean
from learning_dist_metrics.dist_metrics import WeightedDistanceTester


class LDM(object):

    """Learning Distance Metrics (LDM)

    An aglorithm learning distance metrics in a supervsied
    context to tighten spreads of points known similar and
    amplify separation of points considered different in
    a transformed space.

    Attributes:
    -----------
    * trans_vec: array, [n_features, ]
        The A matrix transforming the original datasets

    *ratio: float
        The ratio of sum distances of transformed points known similar
        over its couterpart of tarnsformed points known different
    """

    VERSION = "0.2"

    def __init__(self, solver_method="SLSQP", is_debug=False):

        if not solver_method in ["L-BFGS-B", "SLSQP"]:
            raise ValueError("Only support \"L-BFGS-B\" or \"SLSQP\"!")

        self._solver_method = solver_method
        self._transform_matrix = np.array([])
        self._ratio = 1
        self._is_debug = is_debug

    def fit(self, user_ids, user_profiles, S, D=None, is_directed=False):
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
        is_directed: {boolean},
            False (default), connections formed am undirected graph
            True, connections formed a directed graph

        Returns:
        --------
        _trans_vec: {matrix-like, np.array}, shape(n_features, n_features)
               A transformation matrix (A)
        _ratio: float
        """
        self._fit(user_ids, user_profiles, S, D, is_directed)

    def fit_transform(self, user_ids, user_profiles, S, D=None, is_directed=False):
        """ Fit the model with X, S, D and conduct transformation on X

        Parameters:
        -----------
        X: {matrix-like, np.array}, shape (n_sample, n_features)
            Training data, where n_samples is the number of n_samples
            and n_features is the number of features
        is_directed: {boolean},
            False (default), connections formed am undirected graph
            True, connections formed a directed graph

        Returns:
        --------
        X_new: {marix-like, np.array}, shape (n_sample, n_features)
            The return of X transformed by fitted matrix A
        """
        self.fit(user_ids, user_profiles, S, D, is_directed)
        return self.transform(user_profiles)

    def _data_validator(self, user_profiles, user_ids=None):

        if isinstance(user_profiles, np.ndarray):
            if user_ids is None:
                ids = range(user_profiles.shape[0])
            else:
                ids = user_ids

        elif isinstance(user_profiles, pd.DataFrame):
            col_names = user_profiles.columns.tolist()
            if not (user_ids is None):
                ids = user_ids
            else:
                if 'ID' in col_names or 'id' in col_names:
                    try:
                        ids = user_profiles['ID'].tolist()
                        user_profiles = user_profiles.drop(['ID'], axis=1, inplace=False).as_matrix()
                    except:
                        ids = user_profiles['id'].tolist()
                        user_profiles = user_profiles.drop(['id'], axis=1, inplace=False).as_matrix()
                else:
                    ids = range(user_profiles.shape[0])
                    user_profiles = user_profiles.as_matrix()
        else:
            msg = "user_profiles must be either numpy.ndarray or pandas.DataFrame"
            raise ValueError(msg)

        return ids, user_profiles


    def _fit(self, user_ids, user_profiles, S, D=None, is_directed=False):
        """ Fit the model with given information: user_profiles, S, D

        Fit the learning distance metrics: (1) if only S is given, all pairs of
        items in user_profiles but not in S are considered as in D; (2) if both S and D
        given, items in user_profiles but neither in S nor in D will be removed from
        fitting process.

        Parameters:
        ----------
        user_id: {vector-like}, user
        user_profiles: {matrix-like, np.array}, shape (n_sample, n_features) matrix of
           observations with 1st column keeping observation ID
        S: {vector-like, list} a list of tuples which define a pair of data
           points known as similiar
        D: {vector-like, list} a list of tuples which define a pair of data
           points known as different
        is_directed: {boolean},
            False (default), connections formed am undirected graph
            True, connections formed a directed graph

        Returns:
        --------
        _trans_vec: {matrix-like, np.array}, shape(n_features, n_features)
                    A transformation matrix (A)
        _ratio: float
        """

        user_ids, user_profiles = self._data_validator(user_profiles=user_profiles, user_ids=user_ids)
        n_sample, n_features = user_profiles.shape

        bnds = [(0, None)] * n_features  # boundaries
        init = [1] * n_features  # initial weights

        if D == None:
            all_pairs = [p for p in combinations(user_ids, 2)]

            # generate sampling
            sample_size = 3 * len(S)
            if len(all_pairs) > sample_size:
                samp_idx = choice(range(len(all_pairs)), size=sample_size, replace=False)
                all_pairs = [all_pairs[idx] for idx in samp_idx]

            D = get_exclusive_pairs(all_pairs, S, is_directed)
        else:
            # if D is provided, keep only users being
            # covered either by S or D
            covered_items = get_unique_items(S, D)
            keep_items = [find_index(i, user_ids) for i in user_ids \
                if i in covered_items]
            user_profiles = user_profiles[keep_items, :]

        # Convert user_ids in D and S into row index, in order to provide them to
        # a set of two distance functions, squared_sum_grouped_dist() and
        # sum_grouped_dist()
        S_idx = [(find_index(a, user_ids), find_index(b, user_ids)) for (a, b) in S]
        D_idx = [(find_index(a, user_ids), find_index(b, user_ids)) for (a, b) in D]

        grouped_distance_container = WeightedDistanceTester(user_ids, user_profiles, S_idx, D_idx)

        def objective_func(w):
            return grouped_distance_container.update(w)

        if self._is_debug:
            try:
                print("Examples of S: %s" % S[:5], len(S))
                print("Examples of D: %s" % D[:5], len(D))
                print("Examples of user_profiles: %s" % user_profiles[:5, :], user_profiles.shape)
            except:
                print("Examples of S: %s" % S, len(S))
                print("Examples of D: %s" % D, len(D))
                print("Examples of user_profiles: %s" % user_profiles, user_profiles.shape)

        start_time = time.time()
        fitted = minimize(objective_func, init, method=self._solver_method, bounds=bnds)
        duration = time.time() - start_time

        if self._is_debug:
            print("--- %.2f seconds ---" % duration)

        # optimized value vs. value of initial setting
        try:
            self._transform_matrix = vec_normalized(fitted['x'])
            w = self._transform_matrix
            self._ratio = fitted['fun'] / objective_func(init)
        except:
            self._transform_matrix = init
            w = self._transform_matrix
            self._ratio = None
            msg = ["WARNING!", "information (S) is not sufficient to induce distance leanring metrics! ",
                   "resort to unweigthed distance metrics."]
            print(msg)

        self._dist_func = lambda x, y: weighted_euclidean(x, y, w)

        return self._transform_matrix, self._ratio

    def transform(self, user_profiles):
        """ Tranform user_profiles by the learned tranformation matrix (A)

        Parameters:
        -----------
        user_profiles: {matrix-like, np.array}, shape (n_sample, n_features)
           Training data, where n_samples is the number of n_samples
           and n_features is the number of features

        Returns:
        --------
        X_new: {marix-like, np.array}, shape (n_sample, n_features)
               The return of user_profiles transformed by fitted matrix A
        """
        n_sample, n_features = user_profiles.shape
        trans_matrix = self._transform_matrix

        if len(trans_matrix) != n_features:
            raise ValueError("Transformation matrix is not",
                             "compatiable with user_profiles!")

        return self._transform_matrix * user_profiles

    def get_transform_matrix(self):
        """ Returned the fitted transformation matrix (A)

        Returns:
        -------
        _trans_vec: {matrix-like, np.array}, shape(n_features, n_features)
               A transformation matrix (A)
        """
        return self._transform_matrix

    def fitted_dist_func(self, x, y):
        """ Returned the distance functions used in fitting model

        Returns:
        --------
        func: {function} a function accept (x1, x2, *arg)
        """
        if self._transform_matrix is not None:
            w = self._transform_matrix
            g = lambda x, y: weighted_euclidean(x, y, w)
            return g(x, y)
        else:
            None

    def get_ratio(self):
        """The ratio of aggregate metrics of similiar points
           over the couterparts of different points
        """
        return self._ratio


def get_exclusive_pairs(target_pairs, reference_pairs, is_directed=False):
    """ Remove from target_paris the item (pairs) which
        has matches in reference_pairs.

    Parameters:
    -----------
    target_pairs: {list}, [(1, 2), (1, 3), ...]
    reference_pairs: {list}, [(2, 1), (10, 11)]
    is_directed: {boolean},
        False (default), connections formed am undirected graph
        True, connections formed a directed graph

    Returns:
    -------
    """

    def sorted_pairs_str(a, b):
        if a > b:
            return str(b), str(a)
        else:
            return str(a), str(b)

    def pairs_str(a, b):
        return str(a), str(b)

    if is_directed:
        # directed graph
        target_pairs_str = [" ".join(pairs_str(a, b)) for (a, b) in target_pairs]
        reference_pairs_str = [" ".join(pairs_str(a, b)) for (a, b) in reference_pairs]
    else:
        target_pairs_str = [" ".join(sorted_pairs_str(a, b)) for (a, b) in target_pairs]
        reference_pairs_str = [" ".join(sorted_pairs_str(a, b)) for (a, b) in reference_pairs]

    keep_idx = [ii for ii, signature in enumerate(target_pairs_str) \
                if not (signature in reference_pairs_str)]

    return [target_pairs[idx] for idx in keep_idx]



def get_unique_items(x_pairs, y_pairs):
    """Return all item mentioned either by x_pairs
       or y_pairs.
    """
    x_pairs.extend(y_pairs)
    res = []
    for a, b in x_pairs:
        if a not in res:
            res.append(a)
        if b not in res:
            res.append(b)
    return res


def vec_normalized(x, digits=2):
    """ Noramlize a vector to ensure that the sum of
        elements of the vector equals to one
    """
    x_sum = np.sum(x) * 1.0
    res = [round(i/x_sum, digits) for i in x]
    return res


def find_index(val, array):
    """ Return the index/position of element, whose value equals to val, in
        array
    """
    res = [i for i, v in enumerate(array) if v == val]
    return res[0]
