import unittest
from scipy.spatial.distance import euclidean
from numpy import array
from numpy import sqrt
 
from dist_metrics import weighted_euclidean
from dist_metrics import pairwise_dist_wrapper
from dist_metrics import all_pairwise_dist
from dist_metrics import sum_grouped_dist
from dist_metrics import squared_sum_grouped_dist
 
class test_learing_metric(unittest.TestCase):

    version = "0.2"

    def setUp(self):
        # distance metric test data
        self.x1 = [1, 1, 0, 2]
        self.x2 = [0, 0, 0, 1]
        # grouped_average_distance
        self.data1 = [[4.5, 2], [5.2, 3], [7, 1.5], \
              [2, 3.1], [2.5, 2]]
        self.data1 = array(self.data1)
        self.data1_pairs = [(0, 1), (0, 2), (1, 2), (3, 4)]
 
    def test_unweighted_distnace(self):
        """Test unweighted(or euqally weighted) distance calculation
           which effectively same to standard euclidean distance
        """
        ref_res = euclidean(self.x1, self.x2)
        test_res = weighted_euclidean(self.x1, self.x2)
        self.assertEqual(test_res, ref_res)
 
    def test_weighted_distance(self):
        test_weights = [2, 2, 1, 1]
        scaled_x1 = [ sqrt(w) * v for w, v in zip(test_weights, self.x1) ]
        scaled_x2 = [ sqrt(w) * v for w, v in zip(test_weights, self.x2) ]
        ref_res = euclidean(scaled_x1, scaled_x2)
        test_res = weighted_euclidean(self.x1, self.x2, test_weights)
        self.assertEqual(test_res, ref_res)

    def test_sum_grouped_dist(self):
        """
        """
        cal_res = sum_grouped_dist(self.data1_pairs, self.data1, [1] * 2)
        cal_res = round(cal_res, 2)
        true_res = 7.32
        self.assertEqual(cal_res, true_res)

    def test_squared_sum_grouped_dist(self):
        """
        """
        cal_res = squared_sum_grouped_dist(self.data1_pairs, self.data1, [1] * 2)
        cal_res = round(cal_res, 2)
        true_res = 14.94
        self.assertEqual(cal_res, true_res)
 
 
if __name__ == '__main__':
    unittest.main()
 
 