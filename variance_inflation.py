import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

# Generate some sample data
np.random.seed(0)
n = 100
x1 = np.random.normal(0, 1, n)
x2 = 2*x1 + np.random.normal(0, .1, n)
y = 2 + 2*x1 + x2+ np.random.normal(0, 1, n)
X = np.column_stack((x1, x2))

"""
Randomly pick 90 samples, train linear regression model, and store model weights
"""

# Number of iterations
num_iterations = 200

# Number of samples to pick in each iteration
sample_size = 90

# Store model coefficients
theta = []


# Run iterations
for i in range(num_iterations):
    # Randomly select indices and corresponding samples from X and y
    indices = np.random.choice(range(len(y)), size=sample_size, replace=False)
    selected_X = X[indices,:]
    selected_y = y[indices]

    # Fit linear regressor
    model = LinearRegression().fit(selected_X, selected_y)

    # Store model parameters theta
    theta_0 = model.intercept_
    theta_1 = model.coef_
    theta.append(np.insert(theta_1, 0, theta_0))

    del model

theta = np.array(theta)



plt.figure()
for i in range(theta.shape[1]):
    plt.hist(theta[:,i], bins=50, density=False, label=fr'$\theta$ {i}', alpha=0.7)
plt.xlabel(rf'$\theta$')
plt.ylabel('counts')
plt.legend()
plt.tight_layout()
plt.show()

