import unittest
import numpy as np
from my_r2_score import r2_score as r2

class TestR2Score(unittest.TestCase):

    def test_perfect_pred(self):
        y_true = np.zeros((100,1))
        n = y_true.shape[0]
        for i in range(n):
            y_true[i,0] = np.random.randint(10)
        y_pred = y_true
        result =  r2(y_true, y_pred)
        self.assertAlmostEqual(result, 1.0, delta = 1e-2)

    def test_mean_pred(self):
        y_true = np.zeros((100, 1))
        n = y_true.shape[0]
        for i in range(n):
            y_true[i, 0] = np.random.randint(10)
        y_pred = np.zeros(y_true.shape)
        y_true_mean = y_true.mean()
        for i in range(n):
            y_pred[i, 0]  = y_true_mean
        print(f'type of the y_pred: {type(y_pred)}')
        result = r2(y_true, y_pred)
        self.assertEqual(result, 0.0)

    def test_input_dims(self):
        y_true = np.zeros((99, 1))
        y_pred = np.zeros((100, 1))
        self.assertRaises(ValueError, r2, y_true, y_pred)

    def test_data_type(self):
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        self.assertRaises(ValueError, r2, y_true, y_pred)