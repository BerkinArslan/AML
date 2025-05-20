from exercise01 import lin_regression as linreg
import unittest
import numpy as np
from matplotlib import pyplot as plt

class TestLinRegression(unittest.TestCase):
   def test_regression(self):
       X = np.arange(0, 10, 1)
       Y = X + 0.1 * np.random.randn(10)
       theta_0 , theta_1 = linreg(X, Y)
       self.assertAlmostEqual(theta_0, 0, delta = 0.3)
       self.assertAlmostEqual(theta_1, 1, delta = 0.1)
       plt.plot(X, theta_0 + theta_1 * X, color = 'red')
       plt.show()






if __name__ == "__main__":
     unittest.main()