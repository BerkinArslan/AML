import k_means_ex_03v2 as kmeans
import unittest
import numpy as np

class TestKMeans(unittest.TestCase):
    def test_update_centroids_output(self):
        data = np.array([
            [0.0, 0.1],
            [0.2, -0.1],
            [-0.1, 0.0],
        ])
        labels = np.array([0, 1, 2])
        km = kmeans.KMeans_(k=3, mode='L2')
        centroids = km.update_centroids(data, labels)
        self.assertIsInstance(centroids, np.ndarray)
        self.assertEqual(centroids.shape[1], data.shape[1])

    def test_update_centroids_large_synthetic(self): #chatGPT test
        # -- 1) make three 2D Gaussian clusters around known centers
        rng = np.random.RandomState(42)
        centers = np.array([
            [0.0, 0.0],
            [10.0, 10.0],
            [-10.0, 5.0],
        ])
        points_per_cluster = 100
        data_blocks = []
        label_blocks = []
        for i, c in enumerate(centers):
            # cluster i: normal noise around center c
            block = rng.randn(points_per_cluster, 2) + c
            data_blocks.append(block)
            label_blocks.append(np.full(points_per_cluster, i, dtype=int))

        data = np.vstack(data_blocks)
        labels = np.concatenate(label_blocks)

        # -- 2) compute the exact “true” centroids
        expected = np.vstack([
            data[labels == i].mean(axis=0)
            for i in range(centers.shape[0])
        ])

        # -- 3) run your code
        km = kmeans.KMeans_(k=3, mode='L2')
        result = km.update_centroids(data, labels)

        # -- 4) assertions
        self.assertIsInstance(result, np.ndarray)
        # shape must be (3, 2)
        self.assertEqual(result.shape, expected.shape)
        # numeric match
        self.assertTrue(
            np.allclose(result, expected, atol=1e-8),
            msg=f"expected {expected!r}, got {result!r}"
        )
        #print(expected)
        #print(result)

        #edge_cases

    def test_update_centroids_empty_clusters(self):
        data = np.array([
            [0.0, 0.1],
            [0.2, -0.1],
            [-0.1, 0.0],
        ])
        labels = np.array([0, 1, 1])
        km = kmeans.KMeans_(k=3, mode='L2')
        self.assertRaises(ValueError, km.update_centroids, data, labels)

    def test_update_centroids_empty_data_set(self):
        data = np.array([])
        labels = np.array([])
        km = kmeans.KMeans_(k=3, mode='L2')
        self.assertRaises(ValueError, km.update_centroids, data, labels)

    def test_update_centroids_mismatch_shape(self):
        data = np.array([
            [0.0, 0.1],
            [0.2, -0.1],
            [-0.1, 0.0],
        ])
        labels = np.array([0, 1])
        km = kmeans.KMeans_(k=2, mode='L2')
        self.assertRaises(ValueError, km.update_centroids, data, labels)

    def test_update_centroids_less_data_label_than_k(self):
        data = np.array([
            [0.0, 0.1],
            [0.2, -0.1],
            [-0.1, 0.0],
        ])
        labels = np.array([0, 1])
        km = kmeans.KMeans_(k=3, mode='L2')
        self.assertRaises(ValueError, km.update_centroids, data, labels)

    def test_update_centroids_empty_clusters_with_assigned_centroids(self):
        data = np.array([
            [1.0, 1.1],
            [1.1, 1.0],
            [0.9, 1.0],
            [1.0, 0.9],
            [2.0, 2.1],
            [2.1, 2.0],
            [1.9, 2.0],
            [2.0, 1.9],
            [3.0, 3.1],
            [3.1, 3.0],
            [2.9, 3.0],
            [3.0, 2.9]
        ])
        labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        km = kmeans.KMeans_(k=3, mode='L2')
        km.centroid_locations = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0]
        ])
        centroids = km.update_centroids(data, labels)

        print(labels)
        print(km.centroid_locations)



