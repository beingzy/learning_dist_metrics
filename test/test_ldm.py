import unittest

from pandas import DataFrame

from datasets.load_data import load_sample_data
from learning_dist_metrics.ldm import LDM


class TestClassLDM(unittest.TestCase):
    
    def setUp(self):
        self.ldm = LDM()
        self.data = load_sample_data()

    def test_init_dist_func(self):
        # access default value to test the success of initialization
        self.assertEqual(self.ldm._ratio, 1)
    
    def test_fit(self):

        user_profiles = self.data.data
        user_ids = range(user_profiles.shape[0])
        S = self.data.sim_pairs
        D = self.data.diff_pairs

        self.ldm.fit(user_ids, user_profiles, S, D)

        fitted_transform_matrix = [round(i, 2) for i in self.ldm.get_transform_matrix()]
        estimated_value = [1, 0.0, 0.0]
        self.assertEqual(fitted_transform_matrix, estimated_value)

if __name__ == "__main__":
    unittest.main()
