"""
"""
from os.path import dirname
from pandas import read_csv
from itertools import combinations
from random import shuffle

class Bunch(dict):
    """
    Container object for datasets: dictionary-like object that
    exposes its keys as attributes.
    """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def load_sample_data():
    """
    Load a datasets (nrow: 100, ncol: 4) having inheret clusters.

    ================= =======
    Clusters:               2 
    Samples per class:     50
    Samples total:        100
    Dimensionality:         4
    Features:            real
    ================= =======

    Returns:
    --------
    data: Bunch
        Dictonary-like object, the interesting attributes:
        'data', the data to learn, 'target', the classification labels,
        'traget_names', the meaning of the labels, 'feature_names', the
        meaning of the features, and 'DESCR', the full description of
        the dataset.

    Examples:
    """
    file_path = dirname(__file__)

    data = read_csv(file_path + '/data/sample_2_classes_100.data', header = 0)
    data = data.as_matrix()
    X = data[:, :-1]
    y = data[:, -1]

    #idx_ones = [i for i, val in enumerate(y) if val == 1]
    #idx_zeros = [i for i, val in enumerate(y) if val == 0]
    idx_ones, idx_zeros = [], []
    for i, val in enumerate(y):
        if val == 1:
            idx_ones.append(i)
        if val == 0:
            idx_zeros.append(i)

    sim_pairs, diff_pairs = [], []

    for i in combinations(idx_ones, 2):
        sim_pairs.append(i)
    
    for i in combinations(idx_zeros, 2):
        sim_pairs.append(i)
    
    for a in idx_ones:
        for b in idx_zeros:
            diff_pairs.append((a, b))
    
    shuffle(diff_pairs)

    res = Bunch(
        data = X, target = y, feat_names = ["x1", "x2", "x3"],
        sim_pairs = sim_pairs, diff_pairs = diff_pairs
        )

    return res
