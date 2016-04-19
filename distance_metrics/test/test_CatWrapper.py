"""
"""
import unittest
from pandas import DataFrame
from distance_metrics import GeneralDistanceWrapper


class TestCategoryWrapper(unittest.TestCase):

    def setUp(self):
        x = [1, 2, 'a', 5.5, 'b']
        y = [4, 0, 'a', 2.1, 'c']
        xdf = DataFrame([{'gender': 'female', 'height': 5.6, 'weight': 120},
                         {'gender': 'male', 'height': 5.8, 'weight': 180}])
        self._data = {"x": x, "y": y, "xdf": xdf}
        self._catwrapper = GeneralDistanceWrapper(category_index=[2, 4])

    def test_num_elements(self):
        num_elements, _ = self._catwrapper.decompose(self._data["x"])
        true_num_elements = [1, 2, 5.5]
        self.assertEqual(num_elements, true_num_elements)

    def test_cat_elements(self):
        _, cat_elements = self._catwrapper.decompose(self._data["x"])
        true_cat_elements = ['a', 'b']
        self.assertEqual(cat_elements, true_cat_elements)

    def test_num_component_differnce(self):
        num_diff, cat_diff = self._catwrapper.get_component_difference(self._data["x"], self._data["y"])
        true_num_diff = [-3, 2, 3.4]
        self.assertEqual(num_diff, true_num_diff)

    def test_cat_component_differnce(self):
        num_diff, cat_diff = self._catwrapper.get_component_difference(self._data["x"], self._data["y"])
        true_cat_diff = [0, 1]
        self.assertEqual(cat_diff, true_cat_diff)

    def test_cat_differnce(self):
        res = self._catwrapper.get_difference(self._data["x"], self._data["y"])
        true = [-3, 2, 0, 3.4, 1]
        self.assertEqual(res, true)

    def test_unweighted_euclidean(self):
        # to ensure weight is None in order to perform
        # unweighted euclidean distance calculation
        self._catwrapper.reset_weights()
        dist = self._catwrapper.dist_euclidean(self._data["x"], self._data["y"])
        true_dist = 5.0556898639058154
        self.assertEqual(dist, true_dist)

    def test_weighted_euclidean(self):
        self._catwrapper.load_weights([1, 1, 1, 0, 0])
        dist = self._catwrapper.dist_euclidean(self._data["x"], self._data["y"])
        true_dist = 3.6055512754639891
        self.assertEqual(dist, true_dist)

    def test_convert_method_to_function_unweighted(self):
        self._catwrapper.reset_weights()
        dist_func = self._catwrapper.dist_euclidean
        dist = dist_func(self._data["x"], self._data["y"])
        true_dist = 5.0556898639058154
        self.assertEqual(dist, true_dist)

    def test_convert_method_to_function_weighted(self):
        self._catwrapper.load_weights([1, 1, 1, 0, 0])
        dist_func = self._catwrapper.dist_euclidean
        dist = dist_func(self._data["x"], self._data["y"])
        true_dist = 3.6055512754639891
        self.assertEqual(dist, true_dist)

    def test_fit_list(self):
        self._catwrapper.update_category_index([])
        self._catwrapper.fit(self._data['x'])
        cat_idx = self._catwrapper._cat_idx
        true_cat_idx = [2, 4]
        self.assertEqual(cat_idx, true_cat_idx)

    def test_fit_ndarray(self):
        self._catwrapper.update_category_index([])
        self._catwrapper.fit(self._data['xdf'].as_matrix())
        cat_idx = self._catwrapper._cat_idx
        is_valid_cat_idx = cat_idx == [0]
        self.assertTrue(is_valid_cat_idx)

    def test_fit_dataframe(self):
        self._catwrapper.update_category_index([])
        self._catwrapper.fit(self._data['xdf'])
        all_featnames = self._catwrapper._all_feat_names
        cat_featnames = self._catwrapper._cat_feat_names
        cat_idx = self._catwrapper._cat_idx

        is_valid_all_featnames = all_featnames == ["gender", "height", "weight"]
        is_valid_cat_featnames = cat_featnames == ["gender"]
        is_valid_cat_idx = cat_idx == [0]
        is_succeed = is_valid_all_featnames and is_valid_cat_featnames and is_valid_cat_idx
        self.assertTrue(is_succeed)


if __name__ == '__main__':
    unittest.main()
