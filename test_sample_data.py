from pandas import read_csv
from os.path import dirname
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from itertools import combinations

def get_current_dir():
    """
    return the path where this file resides
    """
    return dirname(__file__)

def get_data_source_dir():
	"""
	return the path of directory storing source data
	"""
	return get_current_dir() + "/datasets/data/"

DATA_PATH = get_data_source_dir()

if __name__ == "__main__":
    try:
    	sample_data = read_csv(DATA_PATH + "sample_2_classes_100.data", header = False)
        print sample_data.head(5)
    except:
    	print "Could not find the file: '{0}'".format(DATA_PATH + "sample_2_classes_100.data")

    # plot the clusters
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection = '3d')
    #ax.scatter(
    #	sample_data.ix[:, 0], sample_data.ix[:, 1], sample_data.ix[:, 2], 
    # 	c = sample_data.ix[:, 3])
    #ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')

    #plt.show()

    idx_ones = [i for i, val in enumerate(sample_data.y) if val == 1]
    idx_zeros = [i for i, val in enumerate(sample_data.y) if val == 0]

    sim_pairs = []
    for i in combinations(idx_ones, 2):
        sim_pairs.append(i)

    for i in combinations(idx_zeros, 2):
        sim_pairs.append(i)

    diff_pairs = []
    for a in idx_zeros:
        for b in idx_ones:
            diff_pairs.append((a, b))

    print "length of similar paris: {0}".format(len(sim_pairs))
    print "length of different paris: {0}".format(len(diff_pairs))








