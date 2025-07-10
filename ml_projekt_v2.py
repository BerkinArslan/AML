import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler



##########     EXPLOTARY DATA ANALYSIS     ##############

def information_from_data_frame(data_frame: pd.DataFrame):
    print(f'Then number of rows: {data_frame.shape[0]}, '
          f' the number of attributes: {data_frame.shape[1]}')
    for col_name in data_frame.columns:
        print(f'Max {col_name}: {max(data_frame[col_name])}, '
              f'Min {col_name}: {min(data_frame[col_name])}')

#cleaning the data:


def clean_the_break_noise_data(data_frame: pd.DataFrame):
    #SPL under 50 is not relevant
    data_frame = data_frame[
        data_frame['max SPL'] > 50
    ]

    #The squel between 0.1 till 23kHz are relevant
    data_frame = data_frame[
        (data_frame['dominant frequency'] > 0.1)&
        (data_frame['dominant frequency'] < 23)
    ]

    #Sensory mistakes in the max break p are not relevant
    data_frame = data_frame[
        data_frame['max break p'] > 0
    ]

    return data_frame
###### Clustering #####
def find_best_silhoutte(data):
    best_eps = 0.05
    best_min_sample = 1
    epss = np.arange(0.05, 1.05, 0.05)
    best_min_samples = np.arange(1, 21, 1)
    sil = -1
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    for eps in epss:
        for min_sample in best_min_samples:
            clusterer = DBSCAN(eps = eps, min_samples=min_sample)
            labels = clusterer.fit_predict(data)
            labels_unique = np.unique(labels)
            if labels_unique.shape[0] < 2:
                pass
            else:
                silhouette = silhouette_score(data, labels)
                if silhouette > sil:
                    sil = silhouette
                    best_eps, best_min_sample = eps, min_sample
                    #print(sil)
    return best_eps, best_min_sample











if __name__ == '__main__':
    data_raw = np.genfromtxt('brake_data_assignment.txt', delimiter=',')

    df_data_raw = {
        'max break p': data_raw[:, 0],
        'v before breaking': data_raw[:, 1],
        'mean steering angle': data_raw[:, 2],
        'dominant frequency': data_raw[:, 3],
        'max SPL': data_raw[:, 4]
    }

    df_data_raw = pd.DataFrame(df_data_raw)

    information_from_data_frame(df_data_raw)

    df_clean_data = clean_the_break_noise_data(df_data_raw)
    print('\n \n')
    information_from_data_frame(df_clean_data)

    for i, column in enumerate(df_clean_data.columns):
        plt.subplot(2, 3, i+1)
        plt.hist(df_clean_data.iloc[:, i], bins = 100, density=True)
        plt.xlabel(f'{column}')
        plt.ylabel('probability')
    plt.tight_layout()
    plt.show()
    # why is the code above generating a plot where the probability for the
    # dominantfrequency is above 1 at some point?

    figure1 = plt.figure(figsize=(10, 8))
    plt.scatter(df_clean_data.iloc[:, 3], df_clean_data.iloc[:, 4], c = 'black')
    plt.xlabel('Frequency kHz')
    plt.ylabel('SPL dB(A)')
    plt.show()

    eps, min_sample = find_best_silhoutte(df_clean_data.iloc[:, 3:4])
    print(eps, min_sample)
    clusterer = DBSCAN(eps = eps, min_samples=min_sample)
    labels = clusterer.fit_predict(df_clean_data.iloc[:, 3:4])

    figure2 = plt.figure(figsize=(10, 8))
    plt.scatter(df_clean_data.iloc[:, 3], df_clean_data.iloc[:, 4], c=labels)
    plt.xlabel('Frequency kHz')
    plt.ylabel('SPL dB(A)')
    plt.show()

    #plotting all of the data with relevant information

    # figure3 = plt.figure(figsize=(10,8))
    # plt.subplot(2,2,1)
    # plt.scatter()
##############################################################
    idx_clean_data_mask = (
        (df_data_raw.iloc[:, 0] > 0)&
        ((df_data_raw['dominant frequency'] > 0.1)&
        (df_data_raw['dominant frequency'] < 23))&
        (df_data_raw['max SPL'] > 50)
    )


    plt.subplot(1,2,2)
    plt.scatter(df_data_raw.loc[~idx_clean_data_mask].iloc[:, 1],
                df_data_raw.loc[~idx_clean_data_mask].iloc[:, 0],
                alpha = 0.5, c = 'gray')
    frequency = df_clean_data.loc[labels == 0].iloc[:, 3].mean()
    plt.scatter(df_data_raw.loc[idx_clean_data_mask].iloc[:, 1].loc[labels == 0],
                df_data_raw.loc[idx_clean_data_mask].iloc[:, 0].loc[labels == 0],
                alpha=0.5, c='green', label=f'f ~ {frequency:.2f} kHz')
    frequency = df_clean_data.loc[labels == 1].iloc[:, 3].mean()
    plt.scatter(df_data_raw.loc[idx_clean_data_mask].iloc[:, 1].loc[labels == 1],
                df_data_raw.loc[idx_clean_data_mask].iloc[:, 0].loc[labels == 1],
                alpha=0.5, c='red',label=f'f ~ {frequency:.2f} kHz')
    frequency =  df_clean_data.loc[labels == 2].iloc[:, 3].mean()
    plt.scatter(df_data_raw.loc[idx_clean_data_mask].iloc[:, 1].loc[labels == 2],
                df_data_raw.loc[idx_clean_data_mask].iloc[:, 0].loc[labels == 2],
                alpha=0.5, c='blue', label=f'f ~ {frequency:.2f} kHz')
    frequency = df_clean_data.loc[labels == 3].iloc[:, 3].mean()
    plt.scatter(df_data_raw.loc[idx_clean_data_mask].iloc[:, 1].loc[labels == 3],
                df_data_raw.loc[idx_clean_data_mask].iloc[:, 0].loc[labels == 3],
                alpha=0.5, c='yellow', label=f'f ~ {frequency:.2f} kHz')
    plt.xlabel(df_data_raw.columns[1])
    plt.ylabel(df_data_raw.columns[0])
    plt.legend(loc='upper left', fontsize='small')
    plt.subplot(1, 2, 1)
    plt.scatter(df_data_raw.loc[~idx_clean_data_mask].iloc[:, 1],
                df_data_raw.loc[~idx_clean_data_mask].iloc[:, 2],
                alpha=0.5, c='gray')
    frequency = df_clean_data.loc[labels == 0].iloc[:, 3].mean()
    plt.scatter(df_data_raw.loc[idx_clean_data_mask].iloc[:, 1].loc[labels == 0],
                df_data_raw.loc[idx_clean_data_mask].iloc[:, 2].loc[labels == 0],
                alpha=0.5, c='green', label=f'f ~ {frequency:.2f} kHz')
    frequency = df_clean_data.loc[labels == 1].iloc[:, 3].mean()
    plt.scatter(df_data_raw.loc[idx_clean_data_mask].iloc[:, 1].loc[labels == 1],
                df_data_raw.loc[idx_clean_data_mask].iloc[:, 2].loc[labels == 1],
                alpha=0.5, c='red', label=f'f ~ {frequency:.2f} kHz')
    frequency = df_clean_data.loc[labels == 2].iloc[:, 3].mean()
    plt.scatter(df_data_raw.loc[idx_clean_data_mask].iloc[:, 1].loc[labels == 2],
                df_data_raw.loc[idx_clean_data_mask].iloc[:, 2].loc[labels == 2],
                alpha=0.5, c='blue', label=f'f ~ {frequency:.2f} kHz')
    frequency = df_clean_data.loc[labels == 3].iloc[:, 3].mean()
    plt.scatter(df_data_raw.loc[idx_clean_data_mask].iloc[:, 1].loc[labels == 3],
                df_data_raw.loc[idx_clean_data_mask].iloc[:, 2].loc[labels == 3],
                alpha=0.5, c='yellow', label=f'f ~ {frequency:.2f} kHz')
    plt.xlabel(df_data_raw.columns[1])
    plt.ylabel(df_data_raw.columns[2])
    plt.legend(loc='upper left', fontsize='small')
    plt.show()









