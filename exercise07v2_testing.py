import unittest
import exercise07v2 as knn
import numpy as np

class TestKNNClassifier(unittest.TestCase):

    def test_init(self):
        classifier = knn.KNNClassifier(3, 'euclidean')
        classifier = knn.KNNClassifier(p = 4)
        classifier = knn.KNNClassifier(metric = 'minkowski')
        self.assertRaises(ValueError, knn.KNNClassifier, 'minkowski')
        self.assertRaises(ValueError, knn.KNNClassifier, metric = 'sss')

    def test_vote(self):
        y = [1, 1, 0, 1, 0 ,0, 1]
        index_array = [0, 1]
        y = np.array(y)
        index_array = np.array(index_array)
        classifier = knn.KNNClassifier()
        self.assertEqual(classifier.vote(index_array, y), 1)
    def test_fit(self):
        pass

    def test_predict(self):
        X = [1, 5, 3, 7, 2, 8]
        y = [0, 1, 0, 1, 0, 0]
        X_query = [6]
        X = np.array(X)
        y = np.array(y)
        X_query = np.array(X_query)
        classifier = knn.KNNClassifier().fit(X, y)
        prediction = classifier.predict(X_query)
        self.assertEqual([1], prediction)

    def test_kfold_split(self):
        x = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        x = np.array(x)
        x = np.tile(x, (3, 1))
        y = [1, 2, 3, 4, 5]
        y = np.array(y)
        y = np.tile(y, 3)
        knn.k_fold_split(x, y, 5, seed = 4665)




