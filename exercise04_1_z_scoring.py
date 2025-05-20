import numpy as np
from matplotlib import pyplot as plt

class Zscorer():
    def __init__(self):
        self.mean_: float | None | np.ndarray = None
        self.std_: float | None | np.ndarray = None
        self.data: np.ndarray | None = None


    def fit(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        :param X: data in array
        :return: mean and the standard deviation of the data
        """
        #checking if the input is in correct form
        if type(X) != np.ndarray:
            raise ValueError('Invalid input type, please give numpy arrays')

        #function
        mean = X.mean(axis = 0)
        std = np.std(X, axis = 0)

        #output
        self.mean_ = mean
        self.std_ = std
        return mean, std


    def transform(self, X: np.ndarray, mean = None, std = None) -> np.ndarray:
        """
        :param X: the data to be normalized with Z-Scoring
        :return: the Normalized data
        """
        #setting up parameter for the function and error handling for param
        if mean is None:
            mean = self.mean_
        if std is None:
            std = self.std_
        if not std.any() or not mean.any():
            raise ValueError('Please first use fit function to calcualte mean and std'
                             'or input mean and std')

        #function
        Norm_X = (1 / std) * (X - mean)

        #output
        return Norm_X




    def inverse_transform(self, X: np.ndarray, mean = None, std = None) -> np.ndarray:
        """
        :param X: the data to be reverted
        :param mean: the mean of the reverted data (should be already a class attribute)
        :param std: the std if the reverted data (should be already a class attribute)
        :return: reverted data
        """
        #setting up parameter or the function and error handling for param
        if mean is None:
            mean = self.mean_
        if std is None:
            std = self.std_
        if std is None or mean is None:
            raise ValueError('Please first use fit function to calcualte mean and std'
                             'or input mean and std')

        #function
        Reverted_X = ((X) * std) + mean

        #output
        return Reverted_X


