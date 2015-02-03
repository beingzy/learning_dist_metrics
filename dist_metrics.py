"""
Collection of Distance Metrics
"""
import numpy as np
import scipy as sp
import itertools as it

def weigthed_euclidean(x, y, w = None):
   """
   Weighted euclidean distance
   """
   if w is None:
       w = [1] * len(x)
   #diff_ = [w_ * (a - b) for a, b, w_ in zip(x, y, w)]
   diff_ = [ (a - b) for a, b in zip(x, y) ]
   weighted_diff_square = [ w * d * d for w, d in zip(w, diff_)]
   return np.sqrt(sum(weighted_diff_square))


def grouped_average_distance(data, id_pairs, weigthts = None):
	"""
	Calculate the grouped distance

	Parameters:
	===========
	data: {matrix-like}, profile data
	id_pairs: {list, }, points msut be same group: 
						[(1, 2), (1, 4), ...]
	weigthts: {vector-like}

	Returns:
	========
	res: {vector}, average distance, size
	"""
	data = np.array(data)

	all_pairs = [i for i in it.combinations(range(len(data)), 2)]
	sim_dist = []
	for p in all_pairs:
		x_ = data[p[0], :]
		y_ = data[p[1], :]
		dist_ = weigthed_euclidean(x_, y_, weigthts)
		if p in id_pairs:
			sim_dist.append(dist_)
		else:
			diff_dist.append(dist_)

	return (np.mean(sim_dist), len(sim_dist))

def bigrouped_average_distance(data, sim_pairs, diff_pairs, weigthts = None):
	"""
	Calculate the grouped distance

	Parameters:
	===========
	data: {matrix-like}, profile data
	sim_pairs: {list, }, points must be same group: 
						[(1, 2), (1, 4), ...]
    diff_pairs: {list, }
	weigthts: {vector-like}

	Returns:
	========
	res: {dict} {"sim_dist": [...], "diff_dist": [...] }
	"""
	data = np.array(data)

	sim_dist, diff_dist = [], []
	for a, b in sim_pairs:
		x_ = data[a, :]
		y_ = data[b, :]
		sim_dist.append(weigthed_euclidean(x_, y_, weigthts))

	for a, b in diff_pairs:
		x_ = data[a, :]
		y_ = data[b, :]
		diff_dist.append(weigthed_euclidean(x_, y_, weigthts))


	n_size_sim = len(sim_dist)
	n_size_diff = len(diff_dist)
	res = {"sim_dist_mean": np.mean(sim_dist), \
		   "sim_size": n_size_sim, \
		   "diff_dist_mean": np.mean(diff_dist), \
		   "diff_size": n_size_diff}
	return res

def min_sim_dist_objective_func(data, id_pairs, dist_weights):
	"""
	"""


def objective_func(data, \
				   id_paris, \
				   dist_weights, \
				   comp_weights = [0.75, 0.25]):
	"""
	Objective function to measure the context to how a new weigths vector 
	perform better than a chosen reference weights. The measure favors the 
	effect of gathering similar points tighter and profound the separation 
	between points not known similar.

	Paramters:
	==========
	data: {matrix-like}, profile data
	id_pairs: {list, }, points msut be same group: 
						[(1, 2), (1, 4), ...]
	dist_weigthts: {vector-like},
	comp_weights: {vector-like}, default value is [0.5, 0.5]

	Returns:
	========
	res: {numeric} weighted metric measures the voerall improvments
	"""

	data = np.array(data)
	ref_weights = [1] * data.shape[1] 
	# reference 
	ref_res = grouped_average_distance(data, id_pairs, ref_weights)
	ref_average_dst_sim = ref_res[0]
	ref_average_dst_diff = ref_res[2]
	# target
	target_res = grouped_average_distance(data, id_pairs, dist_weights)
	target_average_dst_sim = target_res[0]
	target_average_dst_diff = target_res[2]
	# increments
	imp_sim = ref_average_dst_sim - target_average_dst_sim
	imp_diff = target_average_dst_diff - ref_average_dst_diff 
	# sum of weighted increments
	res = imp_sim * comp_weights[0] + imp_diff * comp_weights[1]
	return res

def optimize_fit():
	"""
	"""
	pass







