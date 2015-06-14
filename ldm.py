"""
Learning Distance Metrics Algorithm

Author: Yi Zhang <beingzy@gmail.com>
Create Date: Feb/04/2015
"""
import time
import numpy
import pandas

from itertools import combinations
from scipy.optimize import minimize


from dist_metrics import weighted_euclidean
from dist_metrics import sum_grouped_dist
from dist_metrics import squared_sum_grouped_dist


class LDM(object):

    """Learning Distance Metrics (LDM)

    An aglorithm learning distance metrics in a supervsied
    context to tighten spreads of points known similar and
    amplify separation of points considered different in
    a transformed space.

    Attributes:
    -----------
    _trans_vec: array, [n_features, ]
        The A matrix transforming the original datasets

    _ratio: float
        The ratio of sum distances of transformed points known similar
        over its couterpart of tarnsformed points known different
    """

    VERSION = "0.2"

    def __init__(self, dist_func=None, report_excution_time=True,
                 is_debug=False):
        self._transform_matrix = numpy.array([])
        self._ratio = 1
        self._report_excution_time = report_excution_time
        self._is_debug = is_debug

    def fit(self, X, S, D=None):
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

    def fit_transform(self, X, S, D=None):
        """ Fit the model with X, S, D and conduct transformation on X

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

    def _fit(self, X, S, D=None):
        """ Fit the model with given information: X, S, D

        Fit the learning distance metrics: (1) if only S is given, all pairs of
        items in X but not in S are considered as in D; (2) if both S and D
        given, items in X but neither in S nor in D will be removed from
        fitting process.

        Parameters:
        ----------
        X: {matrix-like, np.array}, shape (n_sample, n_features) matrix of
           observations with 1st column keeping observation ID
        S: {vector-like, list} a list of tuples which define a pair of data
           points known as similiar
        D: {vector-like, list} a list of tuples which define a pair of data
           points known as different

        Returns:
        --------
        _trans_vec: {matrix-like, np.array}, shape(n_features, n_features)
                    A transformation matrix (A)
        _ratio: float
        """
        if isinstance(X, pandas.DataFrame):
            X = X.as_matrix()

        ids = [int(i) for i in X[:, 0]]
        X = X[:, 1:]
        n_sample, n_features = X.shape

        bnds = [(0, None)] * n_features  # boundaries
        init = [1] * n_features  # initial weights

        if D is None:
            all_pairs = [p for p in combinations(ids, 2)]
            D = get_exclusive_pairs(all_pairs, S)
        else:
            covered_items = get_unique_items(S, D)
            X = numpy.delete(X, covered_items, 0)

        # Convert ids in D and S into row index, in order to provide them to
        # a set of two distance functions, squared_sum_grouped_dist() and
        # sum_grouped_dist()
        S_idx = [(find_index(a, ids), find_index(b, ids)) for (a, b) in S]
        D_idx = [(find_index(a, ids), find_index(b, ids)) for (a, b) in D]

        def objective_func(w):
            a = squared_sum_grouped_dist(S_idx, X, w) * 1.0
            b = sum_grouped_dist(D_idx, X, w) * 1.0
            return a - b

        if self._is_debug:
            try:
                print "Examples of S: %s" % S[:5], len(S)
                print "Examples of D: %s" % D[:5], len(D)
                print "Examples of X: %s" % X[:5, :], X.shape
            except:
                print "Examples of S: %s" % S, len(S)
                print "Examples of D: %s" % D, len(D)
                print "Examples of X: %s" % X, X.shape

        start_time = time.time()
        fitted = minimize(objective_func, init, method="L-BFGS-B", bounds=bnds)
        duration = time.time() - start_time

        if self._report_excution_time:
            print("--- %s seconds ---" % duration)

        w = self._transform_matrix
        self._transform_matrix = vec_normalized(fitted['x'])
        # optimized value vs. value of initial setting
        self._ratio = fitted['fun'] / objective_func(init)
        self._dist_func = lambda x, y: weighted_euclidean(x, y, w)

        return (self._transform_matrix, self._ratio)

    def transform(self, X):
        """ Tranform X by the learned tranformation matrix (A)

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
            raise ValueError('Transformation matrix is not \
                compatiable with X!')
        X_new = self._transform_matrix * X
        return X_new

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
    res_pairs = target_pairs[:]
    for ref_pair in reference_pairs:
        for tgt_pair in res_pairs:
            if set(ref_pair) == set(tgt_pair):
                res_pairs.remove(tgt_pair)
                break
    return res_pairs


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
    x_sum = sum(x) * 1.0
    res = [round(i/x_sum, digits) for i in x]
    return res


def find_index(val, array):
    """ Return the index/position of element, whose value equals to val, in
        array
    """
    res = [i for i, v in enumerate(array) if v == val]
    return res[0]

