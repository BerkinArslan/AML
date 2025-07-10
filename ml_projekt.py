import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score

data_original = np.genfromtxt("brake_data_assignment.txt", delimiter=',')
data = np.copy(data_original)
# column names: max break fluid pressure, v before brake stop, mean steering angle, dom f khz, max dba

#for bettter understanding a data frame if ever  needed
pd_data = {
    'max break fluid p': data[:, 0],
    'v before stop': data[:, 1],
    'mean_steering_a': data[:, 2],
    'dominant frequency': data[:, 3],
    'max dB(A)': data[:, 4]
}
df_data = pd.DataFrame(pd_data)


#
# ks = []
# num_of_clusters = []
# for i in range(1, 32):
#     clusterer_ = KMeans(n_clusters=i).fit(normalized_data)
#     clusters = clusterer_.predict(normalized_data)
#     num = np.unique(clusters).shape[0]
#     num_of_clusters.append(num)
#     ks.append(i)
#
# print(ks)
# print(num_of_clusters)
#better: analysis on actual values:
print('max break fluid pressure: ', min(data[:,0]), max(data[:, 0]))
print(f'driving v before breaking: {min(data[:, 1]), max(data[:, 1])}')
print(f'mean steering angle: {min(data[:, 2]), max(data[:, 2])}')
print(f'dominant frequency: {min(data[:, 3]), max(data[:, 3])}')
print(f'maximum dB(a) {min(data[:, 4]), max(data[:, 4])}')

print(f'number of p under zero: {data[data[:, 0] < 0].shape[0]}')
print(f'number of v under zero: {data[data[:, 1] < 0].shape[0]}')
print(f'number of v above 160: {data[data[:, 1] > 160].shape[0]}')
#humand can hear between 1khz to 23khz really good.
print(f'number of kHz under 1: {data[data[:, 2] < 1].shape[0]}')
print(f'number of kHz over 15: {data[data[:, 2] > 15].shape[0]}')
print(f'number of dB(A) under 1: {data[data[:, 3] < 1].shape[0]}')
#most of the data is having a squel under 1khz that why when we ar elooking for outliers
#we should not take dba into considiration
#the same thing goes for frequency range as well.
#steering angles also do look phsically normal.
list_of_vars = ['max break fluid p', 'v before breaking']
print(f'number of kHz under 1: {data[data[:, 2] < 1].shape[0]}')
data = data[:, [0, 1]]
outlier_index = np.array([])
#finding outliers:
#1. with IQR
for i in range(data.shape[1]):
    Q1 = np.percentile(data[:, i], 25)
    Q3 = np.percentile(data[:, i], 75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5  * IQR
    upper_limit = Q3 + 1.5 * IQR

    data_outlier = np.where((data[:, i] < lower_limit) | (data[:, i] > upper_limit))[0]
    #data_outlier = list(data_outlier)
    outlier_index = np.concatenate([outlier_index, data_outlier])
outlier_index = np.unique(outlier_index)
outlier_index = np.array(outlier_index, dtype=int)

#2. with DBSCAN
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)
clusterer = DBSCAN(eps = 0.5, min_samples=10)
clusters = clusterer.fit_predict(normalized_data)
outliers = np.where((clusters == -1))[0]
print(outlier_index.shape[0])
print(outliers.shape[0])

for i in range(data.shape[1]):
    x = data[:, i]
    x_outlier_1 = x[outlier_index]
    x_outlier_2 = x[outliers]
    mask_1 = np.ones(x.shape[0], dtype=bool)
    mask_1[outlier_index] = False
    mask_2 = np.ones(x.shape[0], dtype=bool)
    mask_2[outliers] = False
    #x_1 = x[mask_1]
    x_1 = data[mask_1, i]
    x_2 = x[mask_2]
    for j in range(data.shape[1]):
        if j == i:
            pass
        else:
            figure = plt.figure()
            plt.subplot(1, 2, 1)
            #plt.scatter(x_outlier_1)
            plt.scatter(data[outlier_index, j], x_outlier_1, label = 'outlier')
            plt.scatter(data[mask_1, j], x_1, label = 'not outlier')
            plt.legend()
            plt.xlabel(list_of_vars[j])
            plt.ylabel(list_of_vars[i])
            plt.title('quartile')
            plt.subplot(1,2,2)
            plt.scatter(data[outliers, j], x_outlier_2, label = 'outlier')
            plt.scatter(data[mask_2, j], x_2, label = 'not outlier')
            plt.legend()
            plt.xlabel(list_of_vars[j])
            plt.ylabel(list_of_vars[i])
            plt.title('dbscan')
            figure.show()
#pyhsical outlier detection:

break_p_lower_limit = 0
driving_v_lower_limit = 0
driving_v_upper_limit = 160

data_original = data_original[(data[:, 0] >= 0) & (data[:, 1] <= 160) & (data[:, 1] > 0)]
plt.scatter(data_original[:, 0], data_original[: ,1])
plt.xlabel('break fluid pressure')
plt.ylabel('velocity before breaking')
plt.show()

plt.scatter(data_original[:, 3], data_original[:, 4])
plt.show()
data_original = data_original[((data_original[:, 3] >1) | (data_original[:, 3] == 0))]

scaler = StandardScaler()
data_original_scaled = scaler.fit_transform(data_original)
plt.scatter(data_original_scaled[:, 3], data_original_scaled[:, 4])
plt.show()
model = DBSCAN(eps=0.1, min_samples=5)
#model.fit(data_original_scaled[:, 3:4])
labels = model.fit_predict(data_original_scaled[:, 3:4])

print(np.unique(labels))

plt.scatter(data_original[:, 3], data_original[:, 4], c=labels)
plt.show()

#there is one point that is an outlier detected in a cluster
print(data_original[(data_original[:, 3] < 1) & (data_original[:, 4] > 0)])
# [ 2.82917348e+01  7.66305892e+01  4.73031493e+00  3.86029923e-03
#    6.81314534e+01]]
#this should be out of this data set because humans can not hear 3.8Hz.
#bu I dont want to delete the data with Hz = 0 because it is valueble




#now we will plot only the data with the squel
data_clean = data_original[(labels != -1) & (labels != 0)]
labels_clean = labels[(labels != -1) & (labels != 0)]

figure = plt.figure(figsize=(10, 8))

plt.subplot(2,2,1)
plt.scatter(data_clean[:, 4], data_clean[:, 3], c=labels_clean)
plt.ylabel("Frequency Hz")
plt.xlabel('Sound pressure level dB(A)')
plt.subplot(2,2,2)
plt.scatter(data_clean[:, 0], data_clean[:, 3], c = labels_clean)
plt.xlabel('max break fluid pressure p')
plt.ylabel("Frequency Hz")
plt.subplot(2,2,3)
plt.scatter(data_clean[:, 2], data_clean[:, 3], c = labels_clean)
plt.xlabel('mean steering angle')
plt.ylabel('dominant frequency')
plt.subplot(2,2,4)
plt.scatter(data_clean[:, 1], data_clean[:, 3], c = labels_clean)
plt.xlabel('v before breaking')
plt.ylabel('dominant frequency')
#plt.tight_layout()
figure.suptitle('dependence of frequency')
plt.show()


figure = plt.figure(figsize=(10, 8))

plt.subplot(2,2,1)
plt.scatter(data_clean[:, 3], data_clean[:, 4], c=labels_clean)
plt.xlabel("Frequency Hz")
plt.ylabel('Sound pressure level dB(A)')
plt.subplot(2,2,2)
plt.scatter(data_clean[:, 0], data_clean[:, 4], c = labels_clean)
plt.xlabel('max break fluid pressure p')
plt.ylabel('Sound pressure level dB(A)')
plt.subplot(2,2,3)
plt.scatter(data_clean[:, 2], data_clean[:, 4], c = labels_clean)
plt.xlabel('mean steering angle')
plt.ylabel('Sound pressure level dB(A)')
plt.subplot(2,2,4)
plt.scatter(data_clean[:, 1], data_clean[:, 4], c = labels_clean)
plt.xlabel('v before breaking')
plt.ylabel('Sound pressure level dB(A)')
#plt.tight_layout()
figure.suptitle('dependence of sound pressure')
plt.show()
#this is problematic we will check this...

score_silhoutte = silhouette_score(data_clean, labels_clean)
print(f'The goodness of the clustering is: {score_silhoutte}')
print('A clustering with an average silhouette width of over 0.7 is considered to be "strong", \n'
      'a value over 0.5 "reasonable" and over 0.25 "weak", but with increasing dimensionality \n'
      ' of the data, it becomes difficult to achieve such high values because of the curse of dimensionality, \n'
      ' as the distances become more similar.')


###################################################
#there is a linear dependency to v and sound pressure level
#because p is also dependend on v there is not an important thing to say
#as we can see the cluster with 4khz is relatively overall and accurs onl when there is a high
#breaking pressure or very low one.
#the loudet squels are aornd 6kHz which is to be expected because human ears can hear
#that frequency really good.

#the reason I chose  DBSCAN was that it can see the outliers and is not prototype based.










