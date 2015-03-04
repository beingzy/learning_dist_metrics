import unittest
from scipy.spatial.distance import euclidean
import numpy as np
 
from dist_metrics import weighted_euclidean
from dist_metrics import pairwise_dist_wrapper
from dist_metrics import all_pairwise_dist
from dist_metrics import sum_grouped_dist
from dist_metrics import squared_sum_grouped_dist
 
class test_learing_metric(unittest.TestCase):
 
        def setUp(self):
               # distance metric test data
               self.x1 = [1, 1, 0, 2]
               self.x2 = [0, 0, 0, 1]
               # grouped_average_distance
               self.data1 = [[4.5, 2],
                                     [5.2, 3], 
                                     [7, 1.5], 
                                     [2, 3.1], 
                                     [2.5, 2]]
               self.data1_pairs = [(0, 1), (0, 2), (1, 2), (3, 4)]
 
        def test_unweighted_distnace(self):
               """
               Test unweighted(or euqally weighted) distance calculation
               which effectively same to standard euclidean distance
               """
               ref_res = euclidean(self.x1, self.x2)
               test_res = weighted_euclidean(self.x1, self.x2)
               self.assertEqual(test_res, ref_res)
 
        def test_weighted_distance(self):
               test_weights = [2, 2, 1, 1]
               scaled_x1 = [ np.sqrt(w) * v for w, v in zip(test_weights, self.x1) ]
               scaled_x2 = [ np.sqrt(w) * v for w, v in zip(test_weights, self.x2) ]
               ref_res = euclidean(scaled_x1, scaled_x2)
               test_res = weigthed_euclidean(self.x1, self.x2, test_weights)
               self.assertEqual(test_res, ref_res)
 
        def test_unweighted_grouped_distance(self):
               test_data = np.array(self.data1)
               weights = [1] * test_data.shape[1]
               test_res = grouped_average_distance(test_data, self.data1_pairs, weights)
               test_sim_dist = test_res['sim_dist_mean']
               test_sim_size = test_res['sim_size']
               test_diff_dist = test_res['diff_dist_mean']
               test_diff_size = test_res['diff_size']
               ref_sim_dist, ref_sim_size = 1.8303862046253039, 4
 
               self.assertEqual(test_sim_dist, ref_sim_dist)
               self.assertEqual(test_sim_size, ref_sim_size)
 
 
 
if __name__ == '__main__':
    unittest.main()
 
 