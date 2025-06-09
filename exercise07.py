import numpy as np
import sklearn

class KNNClassifier():
    """
    Hyperparameters:
    metric -> becomes a callable!
    k
    p
    """

    def minkowski_distance(self, p):
        def calculate_distance(a, b):
            a = np.array(a)
            b = np.array(b)
            distance = np.sum(np.abs(a - b) ** p) ** (1/p)
            return distance
        return calculate_distance

    def __init__(self, k: int | float = 3,
                 metric: str | callable = 'euclidean',
                 p: int = None):
        if isinstance(k, int):
            self.k_neighbors = k
            self.frac_neighbors = None
        elif isinstance(k, float):
            self.k_neighbors = None
            self.frac_neighbors = k
        else:
            raise ValueError('k must be an integer or a float')

        if metric == 'euclidean':
            self.metric = self.minkowski_distance(2)
        elif metric == 'minkowski':
            self.metric = self.minkowski_distance(p)
        elif callable(metric):
            self.metric = metric
        else:
            raise ValueError('metric must be {euclidean, minkowski} or a callable')



    def fit(self, X, y):
        #self.data = list(zip(X, y))
        self.data = X
        self.y = y

        if self.frac_neighbors is not None:
            self.k_neighbors = int(np.ceil(X.shape[0] * self.frac_neighbors))
        return self

    def vote(self, knn_index, y):
        votes = y[knn_index]
        number_of_votes = np.bincount(votes)
        majority_vote = np.argmax(number_of_votes)
        return majority_vote

    def predict(self, X_query):
        shortest = []
        for x in X_query:
            shortest_index = []
            all_distances = np.array([self.metric(x, point) for point in self.data])
            for i in range(self.k_neighbors):
                min_index = np.argmin(all_distances)
                shortest_index.append(min_index)
                all_distances[min_index] = np.inf
            shortest.append(shortest_index)
        shortest = np.array(shortest)

        prediction = []
        for vector in shortest:
            vote = self.vote(vector, self.y)
            prediction.append(vote)
        prediction = np.array(prediction)
        return prediction



