import numpy as np
from exercise04_1_z_scoring import Zscorer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
data = np.load('secondary_hand_car_sales.npy', allow_pickle = True)
data_degisti = data [:, -3:]
data_degisti = data_degisti.astype(np.float32)

zscorer = Zscorer()
zscorer.fit(data_degisti)
norm_data = zscorer.transform(data_degisti)
print(norm_data)

scanner = DBSCAN()
clustering = scanner.fit(norm_data)
print(clustering.labels_)

num_clusters = len(np.unique(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
print(num_clusters)

score = silhouette_score(norm_data, clustering.labels_)
print(f"the silhouette score of this clustering is: {score}")

eps_values = np.logspace(-3, 0, num = 40)
eps_values = np.linspace(0.25, 0.8, num=25)
num_of_clusters = []
silhouettes = []
for eps in eps_values:
    scanner = DBSCAN(eps = eps)
    clustering = scanner.fit(norm_data)
    num_clusters = (len(np.unique(clustering.labels_)) -
                    (1 if -1 in clustering.labels_ else 0))
    num_of_clusters.append(num_clusters)
    if num_clusters > 1:
        silhouette = silhouette_score(norm_data, clustering.labels_)
    else:
        silhouette = -1
    silhouettes.append(silhouette)

plt.plot(eps_values, num_of_clusters)
plt.show()
plt.plot(eps_values, silhouettes)
plt.show()

silhouettes = np.array(silhouettes)
sil_max_index = np.argmax(silhouettes)
best_eps = eps_values[sil_max_index]
print(f"the num of clusters in that best epsilon is: {num_of_clusters[sil_max_index]}")
print(best_eps)

colors = ['red', 'green', 'blue', 'black', 'yellow']  # Replace 'white' with 'orange' if visibility is bad
cmap = ListedColormap(colors)

scanner = DBSCAN(eps = best_eps)
clustering = scanner.fit(norm_data)
labels = clustering.labels_
plt.figure()
plt.scatter(data_degisti[:, 0], data_degisti[:, 1], c=labels, cmap = cmap)
plt.colorbar()
plt.xlabel('year of manufacture')
plt.ylabel('mileage')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()
plt.figure()
plt.scatter(data_degisti[:, 0], data_degisti[:, 1], c=labels, cmap = cmap)
plt.colorbar()
plt.xlabel('Brand')
plt.ylabel('Model')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()





