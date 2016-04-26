""" distance matrics
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/22
"""
import numpy as np
from numpy import sqrt
from pandas import DataFrame
from numpy import ndarray


class GeneralDistanceWrapper(object):
    """ Wrapper container to support generalized distance calculation
    for vectors involving both numeric and categorical values

    Example:
    --------
    # caluclate unweighted euclidean dsitance
    x, y =  [1, 2, 'a', 5.5, 'b'], [4, 0, 'a', 2.1, 'c']
    cat_dist_wrapper = GeneralDistanceWrapper()
    cat_dist_wrapper.update_category_index([2, 4])
    xydist = cat_dist_wrapper.dist_euclidean(x, y)

    # weighted euclidean distance
    weights = [1, 1, 1, 0, 0]
    cat_dist_wrapper.load_weights(weights)
    xydist_weighted = cat_dist_wrapper.dist_euclidean(x, y)

    # extract generalized distance calucaltion
    # dist_func is immutable, weights will be automatically
    # update in response to cat_dist_wrapper.load_weights()
    dist_func = cat_dist_wrapper.dist_euclidean
    xydist_weighted = dist_func(x, y)
    """

    def __init__(self, category_index=None, null_val=0.5):
        self._cat_idx = category_index
        self.load_weights()
        self._null_val = null_val

    def fit(self, x):
        """ automate the detection of categoricay variables
        """
        # cat_data_types = [bool, str, object]
        null_val_symbols = [np.nan, None, 'nan', 'null', 'None', 'N/A', '']
        category_dtypes = [str, bool, object, np.str_]

        def return_nonull_values(x):
            nonull_idx = [ii for ii, val in enumerate(x) \
                          if not (val in null_val_symbols)]
            return [x[idx] for idx in nonull_idx]

        def detect_num_vs_cat(val):
            try:
                float(val)
                return False
            except:
                return True

        if isinstance(x, list):
            cat_idx = [ii for ii, val in enumerate(x) if detect_num_vs_cat(val)]

        if isinstance(x, ndarray):
            _, n_feats = x.shape
            cat_idx = []
            for ii in range(n_feats):
                col_vals = x[:, ii]
                col_vals = return_nonull_values(col_vals)
                if len(col_vals) > 0:
                    val = col_vals[0]
                    if detect_num_vs_cat(val):
                        cat_idx.append(ii)

        if isinstance(x, DataFrame):
            _, n_feats = x.shape
            cat_idx = []
            for ii in range(n_feats):
                col_vals = x[:, ii]
                col_vals = return_nonull_values(col_vals)
                if len(col_vals) > 0:
                    val = col_vals[0]
                    if detect_num_vs_cat(val):
                        cat_idx.append(ii)

            all_feat_names = x.columns
            cat_feat_names = [feat_name for ii, feat_name in enumerate(all_feat_names) if ii in cat_idx]
            self.set_features(all_feat_names=all_feat_names, cat_feat_names=cat_feat_names)

        self._cat_idx = cat_idx


    def set_features(self, all_feat_names, cat_feat_names):
        self._all_feat_names = all_feat_names
        self._cat_feat_names = cat_feat_names
        self._cat_idx = [ii for ii, feat in enumerate(all_feat_names) if feat in cat_feat_names]

    def load_weights(self, weights=None, normalize=False):
        if not weights is None:
            if normalize:
                # normalize weights
                sum_weights = sum(weights)
                weights = [w / sum_weights for w in weights]
        self._weights = weights

    def reset_weights(self):
        self._weights = None

    def update_category_index(self, category_index):
        self._cat_idx = category_index

    def get_category_index(self):
        return self._cat_idx

    def decompose(self, x):
        num_elements = [val for ii, val in enumerate(x) if not ii in self._cat_idx]
        cat_elements = [val for ii, val in enumerate(x) if ii in self._cat_idx]
        return (num_elements, cat_elements)

    def recover_vector_from_components(self, num_component, cat_component):
        # output conatainer
        tot_elements = len(num_component) + len(cat_component)
        cat_idx = self._cat_idx
        num_idx = [ii for ii in range(tot_elements) if not ii in cat_idx]
        vector = [None] * tot_elements
        for idx, val in zip(cat_idx, cat_component):
            vector[idx] = val
        for idx, val in zip(num_idx, num_component):
            vector[idx] = val
        return vector

    def get_component_difference(self, a, b):
        if len(a) != len(b):
            raise ValueError("vector (a) is in different size of vector (b)!")
        a_num, a_cat = self.decompose(a)
        b_num, b_cat = self.decompose(b)

        num_diff = [0] * len(a_num)
        for ii, (a, b) in enumerate(zip(a_num, b_num)):
            if _is_null_value(a) or _is_null_value(b):
                num_diff[ii] = self._null_val
            else:
                num_diff[ii] = a - b

        cat_diff = [0] * len(a_cat)
        for ii, (a, b) in enumerate(zip(a_cat, b_cat)):
            if _is_null_value(a) or _is_null_value(b):
                cat_diff[ii] = self._null_val
            else:
                if a == b:
                    cat_diff[ii] = 0
                else:
                    cat_diff[ii] = 1

        return (num_diff, cat_diff)

    def get_difference(self, a, b):
        num_diff, cat_diff = self.get_component_difference(a, b)
        return self.recover_vector_from_components(num_diff, cat_diff)

    def dist_euclidean(self, a, b):
        """ calculate the weighted euclidean distance
        """
        diff = self.get_difference(a, b)
        if self._weights is None:
            return sqrt(sum([val * val for val in diff]))
        else:
            if len(self._weights) == len(diff):
                return sqrt(sum([w * val * val for w, val in zip(self._weights, diff)]))
            else:
                raise ValueError("weights must be in same size with input vector (a, b)!")


def _is_null_value(x):
    null_val_symbols = [np.nan, None, '', 'N/A']
    if x in null_val_symbols:
        return True
    else:
        return False
