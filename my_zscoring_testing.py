import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from exercise04_1_z_scoring import Zscorer
data = np.genfromtxt("data_clustering.csv", delimiter=',')

class TestZscoring(unittest.TestCase):
    def test_fit_results(self):
        zs = Zscorer()
        zs.fit(data)
        scaler = StandardScaler()
        scaler.fit(data)
        self.assertTrue(np.allclose(zs.mean_, scaler.mean_, atol = 1e-3))

    def test_transform_results(self):
        zs = Zscorer()
        skzs = StandardScaler()
        zs.fit(data)
        skzs.fit(data)
        self.assertTrue(np.allclose(zs.transform(data), skzs.transform(data),  atol = 1e-3))

    def test_inverse_transform_results(self):
        zs = Zscorer()
        skzs = StandardScaler()
        zs.fit(data)
        skzs.fit(data)
        Norm_X = zs.inverse_transform(zs.transform(data))
        sk_Norm_X = skzs.inverse_transform(skzs.transform(data))
        self.assertTrue(np.allclose(Norm_X, sk_Norm_X, atol = 1e-3))






