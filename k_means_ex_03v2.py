import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

class KMeans_():
    def __init__(self, centroid_locations: np.ndarray = np.array([], dtype = float),
                assigned_labels_to_the_point: np.ndarray = np.array([], dtype = float),
                unique_labels: np.ndarray | list = [],
                k: int = None, mode : str = 'L2'):
        self.centroid_locations: np.ndarray = centroid_locations
        self.assigned_label_to_the_point = assigned_labels_to_the_point
        self.unique_labels = unique_labels
        self.k = k
        self.mode = mode

    def update_centroids(self, data: np.ndarray, data_label: np.ndarray) -> np.ndarray:
        """
        :param data: data points N data points and n attributes for that data (N,n)
        :param data_label: for every data point the assigned label (N, 1)
        :return: the list of positions for centroids (k, n)
        """
        if data.shape[0] != data_label.shape[0]:
            raise ValueError('data and labels should have the same number of rows')
        unique_labels = np.unique(data_label)
        centroids = []
        if unique_labels.shape[0] != self.k and self.centroid_locations.shape[0] != self.k:
            raise ValueError('This function updates the existing centroid locations\n'
                             'it does not create new clusters.\n'
                             'there should be clusters in the data.')
        if unique_labels.shape[0] != self.k:
            self.relocate_empty_cluster(data_label, data)
            unique_labels = np.unique(self.labels) #burada sorun bos array olusturuyo
        for label in unique_labels:
            data_from_that_cluster = data[data_label == label]
            if data_from_that_cluster.shape[0] != 0:
                match self.mode:
                    case 'L1':
                        centroid_position = np.median(data_from_that_cluster, axis = 0)
                    case 'L2':
                        centroid_position = data_from_that_cluster.mean(axis = 0)
                    case _:
                        raise ValueError('invalid mode')
            else:
                self.relocate_empty_cluster(data_label, data) #should I pass only one label or all of the labels?
            centroids.append(centroid_position)
        self.centroid_locations = np.array(centroids)
        return self.centroid_locations

    def assign_cluster(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        labels = []
        for point in data:
            closest = centroids[0]
            closest_idx = 0
            match self.mode:
                case 'L1':
                    distance = (abs(point - closest)).sum()
                case 'L2':
                    distance = ((point - closest) ** 2).sum()
                case _:
                    raise ValueError('invalid mode')
            for i, centroid in enumerate(centroids):
                match self.mode:
                    case 'L1':
                        temp_distance = abs(point - centroid).sum()
                    case 'L2':
                        temp_distance = ((point - centroid) ** 2).sum()
                    case _:
                        raise ValueError('invalid mode')
                if temp_distance < distance:
                    distance = temp_distance
                    closest = centroid
                    closest_idx = i
            labels.append(closest_idx)
        labels = np.array(labels)
        self.labels = labels
        return self.labels

    def relocate_empty_cluster(self, labels, data: np.ndarray):
        """
        :param labels: labels assigned to every data point
        :param data: data points
        :return: new list of centroid locations (not yet assigned)
        """
        number_of_assigned_labels = np.unique(labels).shape[0]
        number_of_all_labels = self.k
        if number_of_assigned_labels != number_of_all_labels:
            #find the labels that are not assigned to any data point
            empty_labels = [i for i in range(number_of_all_labels) if i not in labels]
            if self.centroid_locations.shape[0] == self.k:
                empty_centroids = []
                for empty_label in empty_labels:
                    #find the centroid location of the empty cluster
                    empty_centroid = self.centroid_locations[empty_label]
                    empty_centroids.append(empty_centroid)
            else:
                raise ValueError('invalid centroid locations')
        else:
            return
        empty_centroids = np.array(empty_centroids)
        #relocate the centroid of the empty cluster to a data point
        N = data.shape[0]
        for empty_label, empty_centroid in enumerate(empty_centroids):
            random_index = np.random.randint(0, N )
            while np.any(np.all(self.centroid_locations == data[random_index], axis = 1)):
                random_index = np.random.randint(0, N)
            empty_centroid = data[random_index]
            self.centroid_locations[empty_label] = empty_centroid
        #assign data points to the new cluster centroid
        self.assign_cluster(data, self.centroid_locations)
        return self.centroid_locations




if __name__ == '__main__':

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


    km = KMeans_(k = 3, mode = 'L2')
    centroids = km.update_centroids(data, labels)
    print('Centroids:', centroids)
    labels = km.assign_cluster(data, centroids)
    print('Assigned labels:', labels)







