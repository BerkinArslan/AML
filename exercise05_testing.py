import unittest

from exercise05 import Node
import numpy as np
from unittest import TestCase

class TestBinaryDecisionTree(unittest.TestCase):

    #testing the entropy first
    def test_entropy_edge_case_1_empty_array(self):
        y = np.array([])
        tree = Node()
        self.assertEqual(tree.entropy(y), 0.0)
    def test_entropy_edge_case_2_pure_population(self):
        y = np.array([0, 0, 0, 0])
        tree = Node()
        self.assertEqual(tree.entropy(y), 0.0)
        y = np.array([1])
        self.assertEqual(tree.entropy(y), 0.0)
    def test_entropy_max(self):
        y = np.array([1, 1, 0, 0])
        tree = Node()
        self.assertEqual(tree.entropy(y), 1.0)

    #testing the edge cases for information gain
    def test_information_gain_edge_case_1_shape_mismatch(self):
        y = np.array([1, 0, 1])
        index_split = np.array([1, 0])
        tree = Node()
        self.assertRaises(ValueError, tree.information_gain, y, index_split )

    def test_information_gain_edge_case_2_empty_arrays(self):
        y = np.array([])
        index_split = np.array([])
        tree = Node()
        self.assertEqual(tree.information_gain(y, index_split), 0.0)
    def test_information_gain_edge_case_3_invalid_index_split(self):
        y = np.array([0, 1, 0, 1, 0, 0])
        index_split = np.array([1, 0, 1, 1, 0, 2])
        tree = Node()
        self.assertRaises(ValueError, tree.information_gain, y, index_split)
    def test_best_split_edge_case_1_constant_feature(self):
        x = np.array([[1], [1], [1], [1]])
        y = np.array([0, 1, 0, 1])
        tree = Node()
        split = tree.best_split(x, y, split = True)
        self.assertTrue(np.all(split == 1) or np.all(split == 0))

    def test_best_split_edge_case_2_all_same_label(self):
        x = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 0, 0])
        tree = Node()
        split = tree.best_split(x, y, split = True)
        self.assertTrue(np.all(split == 1) or np.all(split == 0))

    def test_best_split_edge_case_3_empty_inputs(self):
        x = np.empty((0, 2))
        y = np.array([])
        tree = Node()
        split = tree.best_split(x, y, split = True)
        self.assertEqual(split.size, 0)

    def test_best_split_edge_case_4_single_sample(self):
        x = np.array([[5]])
        y = np.array([1])
        tree = Node()
        split = tree.best_split(x, y, split = True)
        self.assertEqual(split.shape[0], 1)
        self.assertIn(split[0], (0, 1))

    def test_best_split_edge_case_5_multiple_optimal_splits(self):
        x = np.array([[1], [2], [3], [4]])
        y = np.array([0, 1, 0, 1])
        tree = Node()
        split = tree.best_split(x, y, split = True)
        self.assertEqual(split.shape[0], 4)
        self.assertTrue(set(split).issubset({0, 1}))

    def test_best_split_with_the_data(self):
        data = np.genfromtxt("decision_tree_dataset.txt", delimiter=',')
        # print(data)
        x = data[:, 0:2]
        y = data[:, -1]
        tree = Node()
        best_split = tree.best_split(x, y, split = True)
        #print(best_split)

        self.assertEqual(best_split.shape[0], x.shape[0])
    def test_create_split_basic(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        tree = Node()
        self.assertTrue(np.array_equal(tree.create_split(x, 0, 3), np.array([1, 1, 0])))

    def test_create_split_equal_values(self):
        x = np.array([[1, 1], [1, 1], [1, 1]])
        tree = Node()
        self.assertTrue(np.array_equal(tree.create_split(x, 1, 1), np.array([1, 1, 1])))

    def test_create_split_invalid_dim_negative(self):
        x = np.array([[1, 2], [3, 4]])
        tree = Node()
        self.assertRaises(ValueError, tree.create_split, x, -1, 2)

    def test_create_split_invalid_dim_too_large(self):
        x = np.array([[1, 2], [3, 4]])
        tree = Node()
        self.assertRaises(ValueError, tree.create_split, x, 2, 2)

    def test_create_split_float_threshold(self):
        x = np.array([[0.5, 1.2], [1.3, 2.4], [1.1, 1.8]])
        tree = Node()
        self.assertTrue(np.array_equal(tree.create_split(x, 0, 1.2), np.array([1, 0, 1])))





