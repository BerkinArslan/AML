import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

class KMeans_():
    def __init__(self, labels = np.ndarray([])):
        self.centroid_locations: np.ndarray = np.ndarray([])
        self.labels: np.ndarray = labels
        self.all_labels = np.array([], dtype= int)
    def update_centroids(self, data: np.ndarray, data_label: np.ndarray, mode: str = 'L2') -> np.ndarray:
        centroids = []
        self.all_labels = np.array(self.all_labels)
        if self.all_labels.size == 0:
            self.all_labels = np.unique(data_label)
            unique_labels = self.all_labels
        else:
            unique_labels = self.all_labels
        for label in unique_labels:
            positions_from_that_label = data[data_label == label]
            if mode == "L1":
                if positions_from_that_label.size == 0:
                    centroids.append(None)
                    continue
                else:
                    centroid = np.median(positions_from_that_label, axis=0)
            elif mode == "L2":
                if positions_from_that_label.size == 0:
                    centroids.append(None)
                    continue
                else:
                    centroid = positions_from_that_label.mean(axis = 0)
            else:
                raise ValueError('Invalid mode')
            centroids.append(centroid)
        self.centroid_locations = centroids
        centroids = self.relocate_empty_centroid(unique_labels)
        self.centroid_locations = centroids
        centroids = np.array(centroids)

        return centroids

    def assign_cluster(self, data: np.ndarray, centroids: np.ndarray, mode: str = 'L2') -> np.ndarray:
        #labels = np.empty((0, 1))
        labels = []
        for point in data:
            closest = centroids[0]
            closest_idx = 0
            if mode == 'L2':
                distance = ((point - closest) ** 2).sum()
            elif mode == 'L1':
                distance = (abs(point - closest)).sum()
            else:
                raise ValueError('Invalid mode')
            for i,  centroid in enumerate(centroids):
                if mode == 'L2':
                    temp_distance = ((point - centroid) ** 2).sum()
                elif mode == 'L1':
                    temp_distance = (abs(point - centroid)).sum()
                else:
                    raise ValueError('Invalid mode')
                if temp_distance < distance:
                    distance = temp_distance
                    closest = centroid
                    closest_idx = i
            #labels = np.vstack((labels,  closest_idx))
            labels.append(closest_idx)
        #labels = labels[: ,0]
        labels = np.array(labels)
        labels = labels.astype(int)
        self.labels = labels
        return labels

    def is_converged(self, old_centroids: np.ndarray, new_centroids: np.ndarray, tol: float = 1e-1) -> bool:
        is_converged_flag = True
        for old_centroid, new_centroid in zip(old_centroids, new_centroids):
            if ((old_centroid - new_centroid) ** 2).sum() > tol:
                is_converged_flag = False
        return is_converged_flag

    def kmeans_clustering(self, data: np.ndarray, K: int,
                          mode: str = 'L2',
                          init_centroids: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        all_labels = []
        for i in range(K):
            all_labels.append(i)
        self.all_labels = all_labels
        self.labels = all_labels
        if init_centroids == None:
            random_indexes = np.random.randint(0, data.shape[0], K)
            init_centroids = []
            for index in random_indexes:
                init_centroids.append(data[index])
            init_centroids = np.array(init_centroids)
        trial = 0
        data_label = self.assign_cluster(data, init_centroids)
        if trial == 0 and mode == 'L2':
            old_centroids = init_centroids
            new_centroids =  self.update_centroids(data, data_label, 'L2')
            labels = self.assign_cluster(data, new_centroids, 'L2')
            trial = 1
        elif trial == 0 and mode == 'L1':
            old_centroids = init_centroids
            new_centroids = self.update_centroids(data, data_label, 'L1')
            labels = self.assign_cluster(data, new_centroids, 'L1')
            trial = 1
        if trial > 0 and mode == 'L2':
            old_centroids = init_centroids
            new_centroids = self.update_centroids(data, data_label, 'L2')
            labels = self.assign_cluster(data, new_centroids, 'L2')
            while trial < 500 and not (self.is_converged(old_centroids, new_centroids)):
                old_centroids = new_centroids
                new_centroids = self.update_centroids(data, labels, mode = 'L2')
                labels = self.assign_cluster(data, labels, mode = 'L2')
                print(self.sse(data, new_centroids, labels, mode='L2'))
                print(self.bss(data, new_centroids, labels, mode='L2'))
                trial = trial + 1
        if trial > 0 and mode == 'L1':
            old_centroids = init_centroids
            new_centroids = self.update_centroids(data, data_label, 'L1')
            labels = self.assign_cluster(data, new_centroids, 'L1')
            while trial < 500 and not (self.is_converged(old_centroids, new_centroids)):
                old_centroids = new_centroids
                new_centroids = self.update_centroids(data, labels, mode = 'L1')
                labels = self.assign_cluster(data, labels, mode = 'L1')
                print(self.sse(data, new_centroids, labels, mode = 'L1'))
                print(self.bss(data, new_centroids, labels, mode = 'L1'))
                trial = trial + 1
        return new_centroids, labels

    """def relocate_empty_centroid(self, unique_labels):
        empty_labels = [(i, label) for i, label in enumerate(unique_labels) if label not in self.labels]
        if len(empty_labels) != 0:
            for empty_label_index, _ in empty_labels:
                random_index = np.random.randint(0, data.shape[0])
                self.centroid_locations = np.array(self.centroid_locations)
                cand = data[random_index]
            if not any(np.array_equal(cand, row) for row in self.centroid_locations):
                self.centroid_locations[empty_label_index] = cand
        centroids = self.centroid_locations
        return centroids"""

    def relocate_empty_centroid(self, unique_labels):
        # Find cluster‐IDs (0…K-1) that got no points
        empty_labels = [
            (i, label)
            for i, label in enumerate(self.all_labels)
            if label not in unique_labels
        ]
        if empty_labels:
            # Ensure centroids is an ndarray
            self.centroid_locations = np.array(self.centroid_locations)
            for empty_label_index, _ in empty_labels:
                # Pick a random data point as candidate
                random_index = np.random.randint(0, data.shape[0])
                cand = data[random_index]

                # Only reseed if cand isn't already one of the centroids
                if not any(np.array_equal(cand, row)
                           for row in self.centroid_locations):
                    self.centroid_locations[empty_label_index] = cand

        return self.centroid_locations

    def sse(self, data: np.ndarray, centroid_locations: np.ndarray,
                                labels: np.ndarray, mode: str = 'L2') -> float:
        unique_labels = np.unique(labels)
        if data.shape[0] == 0:
            return
        if centroid_locations.shape[0] == 0:
            return
        if labels.shape[0] == 0:
            return
        sse = 0
        for i, label in enumerate(unique_labels):
            points_from_that_cluster = data[labels == label]
            centroid = centroid_locations[i]
            for point in points_from_that_cluster:
                if mode == 'L1':
                    sse_temp = abs(point - centroid).sum()
                    sse_temp = sse_temp ** 2
                elif mode == 'L2':
                    sse_temp = ((point - centroid) ** 2).sum() ** 0.5
                    sse_temp = sse_temp ** 2
                else:
                    raise ValueError('Mode not valid')
                sse = sse + sse_temp
        return sse

    def bss (self, data: np.ndarray, centroid_locations: np.ndarray,
                                labels: np.ndarray, mode: str = 'L2') -> float:
        if mode == 'L1':
            x_mean = np.median(data)
        elif mode == 'L2':
            x_mean = data.mean()
        else:
            raise ValueError('Invalid mode')
        bss = 0
        for centroid in centroid_locations:
            if mode == 'L1':
                bss = bss + abs(x_mean - centroid).sum() ** 2
            elif mode == 'L2':
                bss = bss + ((x_mean - centroid) ** 2).sum()
        return bss





#bgarbage collection

"""class KMeans:
    def __init__(self):
        # actually attach these as attributes!
        self.centroid_locations: np.ndarray = None
        self.labels:           np.ndarray = None
        self.training_data_set: np.ndarray = None

    def update_centroids(
        self,
        data: np.ndarray,
        data_label: np.ndarray,
        mode: str = 'L2'
    ) -> np.ndarray:
        unique_labels = np.unique(data_label)
        # For each label, select all rows with that label and compute the mean.
        centroids = [
            data[data_label == lbl].mean(axis=0)
            for lbl in unique_labels
        ]
        # Stack into shape (n_clusters, n_features)
        self.centroid_locations = np.vstack(centroids)
        return self.centroid_locations"""

"""def update_centroids(self, data: np.ndarray, data_label: np.ndarray, mode: str = 'L2'):
    unique_labels = np.unique(data_label)
    centroid_positions = []
    for label in unique_labels:
        positions_from_that_label  = np.zeros((0, data.shape[1]))
        for i in range(data.shape[0]):
            if data_label[i] == label:
                positions_from_that_label = np.vstack((positions_from_that_label, data[i, :]))
        centroid_position = (positions_from_that_label.sum(axis = 0)
                             / labels[labels[:] == label].shape[0])
        centroid_positions.append(centroid_position)
    self.centroid_locations = centroid_positions
    return self.centroid_locations"""


# 3 clusters × 3 points = 9 total points in 2D
data = np.array([
    # Cluster 0
    [ 0.0,  0.1],
    [ 0.2, -0.1],
    [-0.1,  0.0],
    # Cluster 1
    [ 5.0,  5.0],
    [ 5.1,  4.9],
    [ 4.8,  5.2],
    # Cluster 2
    [10.0,  0.0],
    [ 9.8,  0.2],
    [10.1, -0.1],
])

# Labels: three 0’s, three 1’s, three 2’s
labels = np.array([
    0, 0, 0,
    1, 1, 1,
    2, 2, 2
])

print("data:\n", data)
#print("labels:", labels)

# Now run your centroid update
km = KMeans_(labels)
centroids = km.update_centroids(data, labels)
print("Estimated centroids:\n", centroids)
labels = km.assign_cluster(data, centroids)
print("Labels:\n", labels)
centroids, labels = km.kmeans_clustering(data, 3, mode = 'L2')
print("Labels:\n", labels)
print('Centroids: \n', centroids)

#checking with sklearn
kmeans = KMeans(3).fit(data)
print(kmeans.cluster_centers_)


# fix RNG so the “random reseed” is reproducible
np.random.seed(42)

# Toy data: two true clusters around 0 and 10
data = np.array([[ 0.0],
                 [ 0.2],
                 [-0.1],
                 [10.0],
                 [ 9.8],
                 [10.1]])

# Run KMeans_ with K=3
km = KMeans_()
centroids, labels = km.kmeans_clustering(data, K=3, mode='L2')

print("Final centroids:", centroids.flatten())
print("Final labels:   ", labels)

import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt

# 1) Generate a larger synthetic dataset:
#    - 500 samples
#    - 5 true centers in 2D
#    - varying cluster spreads
data, true_labels = make_blobs(
    n_samples=500,
    centers=[(-10, -10), (-10, 10), (0, 0), (10, -10), (10, 10)],
    cluster_std=[1.0, 2.0, 1.5, 1.0, 2.5],
    random_state=42
)

# 2) Visualize it (optional)
plt.scatter(data[:,0], data[:,1], c=true_labels, s=10, cmap='tab10')
plt.title("Synthetic Blobs (500 pts, 5 clusters)")
plt.show()

# 3) Run your KMeans_
km = KMeans_()
# Tell it we want K=5 clusters
centroids, labels = km.kmeans_clustering(data, K=5, mode='L2')

# 4) Plot the result
plt.scatter(data[:,0], data[:,1], c=labels, s=10, cmap='tab10')
plt.scatter(centroids[:,0], centroids[:,1], c='k', marker='x', s=100)
plt.title("Your KMeans_ Result")
plt.show()

# 5) Inspect final metrics
print("Final SSE:", km.sse(data, centroids, labels, mode='L2'))
print("Final BSS:", km.bss(data, centroids, labels, mode='L2'))




