#Exercize Lineat Regression and Test Driven Development
import sklearn
import numpy as np
from matplotlib import pyplot as plt


def lin_regression(X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    NormalMat = np.zeros((2,2))
    N = X.shape[0]
    NormalMat[0,0] = N
    NormalMat[1,0] = sum(element for element in X)
    NormalMat[0,1] = np.sum(X)
    NormalMat[1,1] = np.sum(X ** 2)

    C = [[np.sum(Y)],[np.sum(Y * X)]]

    Theta = np.linalg.solve(NormalMat, C)
    theta_0 = Theta[0, 0]
    theta_1 = Theta[1, 0]

    print(f"Theta_0 = {theta_0}, Theta_1 = {theta_1}")

    return (theta_0, theta_1)




"""
def test_lin_regression():
    # Simple test data where we know the relationship: y = 2x + 1
    X_test = np.array([0, 1, 2, 3, 4])
    Y_test = 2 * X_test + 1

    # Run your lin_regression function
    theta_0, theta_1 = lin_regression(X_test, Y_test)

    print("\nTesting lin_regression...")
    print(f"Expected intercept (theta_0): 1, found: {theta_0}")
    print(f"Expected slope (theta_1): 2, found: {theta_1}")

    # Check if close enough (allow tiny floating-point errors)
    assert np.isclose(theta_0, 1, atol=1e-6), "theta_0 is incorrect"
    assert np.isclose(theta_1, 2, atol=1e-6), "theta_1 is incorrect"

    print("Test passed!")

# Now you can call it
test_lin_regression()
"""



