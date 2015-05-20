import unittest
import numpy as np
from ldm import LDM
from datasets import load_sample_data


class TestClassLDM(unittest.TestCase):
    
    def setUp(self):
    	self.ldm = LDM()
    	self.data = load_sample_data()
      self.init_ldm = self.ldm

   	def test_init_dist_func(self):
   		ldm = self.init_ldm
   		self.assertEqual(ldm.get_dist_func(), None)
   		self.assertEqual(ldm.get_transform_matrix(), np.array([]))
   		self.assertEqual(ldm.get_ratio(), 1)
    
    def test_fit(self):
    	X = self.data.data 
    	S = self.data.sim_pairs
    	D = self.data.diff_pairs
    	self.ldm.fit(X, S, D)
    	fitted_transform_matrix = [round(i,2) for i in self.ldm.get_transform_matrix()]
        estimated_value = [4.51, 0.0, 0.0]
    	self.assertEqual(fitted_transform_matrix, estimated_value)

if __name__ == "__main__":
    unittest.main()
