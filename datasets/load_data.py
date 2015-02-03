"""
"""

class Bunch(dict):
    """
    Container object for datasets: dictionary-like object that
    exposes its keys as attributes.
    """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def get_data_home(data_home=None):
    """
    Return the path of data dir of this module.

    By default the data dir is set to a folder named 'data'
    in the user home folder
	"""
	pass


def load_2clusters_sample100():
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
	module_path = dirname(__file__)
	pass