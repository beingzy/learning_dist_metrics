import unittest
from load_data import load_sample_data

class TestDataLoader(unittest.TestCase):

    def setUp(self):
	    self.sample_data = load_sample_data()
    
    def test_data_shape(self):
    	data = self.sample_data.data
        n_sample, n_features = data.shape 
        self.assertEqual(n_sample, 100)
        self.assertEqual(n_features, 3)

if __name__ == "__main__":
    unittest.main()