import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix

#######     Data Preprocessing     ########
df_raw = pd.read_csv("ai4i2020.csv")
pd.set_option('display.max_columns', None)
print(df_raw)

def calculate_stats_columns(df):
    """
    this function will calculate the standard deviations and ranges for the
    given columns to see what to see as outlier
    :param df: the dataset to investigate
    :return: terminal print with the information needed about the value ranges and std
    """
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
            min_col = np.min(df[column])
            max_col = np.max(df[column])
            std_col = np.std(df[column])
            mean_col = np.mean(df[column])
            print(f'{column}: mean value: {mean_col}    '
                  f'value range: {min_col} - {max_col}    '
                  f'standard deviation: {std_col}')
        except ValueError:
            print(f'{column} is non numerical')

calculate_stats_columns(df_raw)

"""
Now we decide which columsn are useful for us
I think every numerical value here is very important
so we will take every numerical value
there is another problem
the data is very skewed meaning there are much more not faulty machines than faulty ones
#for the one column that is not muerical but also important: model type.
we can set integer valeus: L = 0, M = 1, L =2
"""

#lets update the weights:
n_0 = np.sum(df_raw['Machine failure'] == 0)
n_1 = np.sum(df_raw['Machine failure'] == 1)
n_total = df_raw.shape[0]
weight_0 = 1
weight_1 = (n_0/n_total) * (n_total/n_1)
weights = []
for status in df_raw['Machine failure']:
    if status == 0:
        weights.append(weight_0)
    elif status == 1:
        weights.append(weight_1)


y = df_raw.iloc[:, -6:]
y = np.array(y)
print(y)
X = df_raw.iloc[:, 2:-6]
X = np.array(X)

print(X[:, 0])
for i, type in enumerate(X[:, 0]):
    if type == 'L':
        X[i, 0] = 0
    elif type == 'M':
        X[i, 0] = 1
    elif type == 'H':
        X[i, 0] = 2

print(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)

folding_machine = KFold()
folds = folding_machine.split(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)


maxes_list = [2, 4, 8, 16, 32, 64, 128, 256]
plt.figure(figsize=(15, 8))
for i in range(6):
    print(f'failure type {i}')
    rf_recalls = []
    dt_recalls = []
    f1_scores_dt = []
    f1_scores_rf = []
    for max in [2, 4, 8, 16, 32, 64, 128, 256]:
        DT = DecisionTreeClassifier(max_depth=max, class_weight='balanced')
        RF = RandomForestClassifier(n_estimators=max, class_weight='balanced')
        DT.fit(X_train, y_train)
        RF.fit(X_train, y_train)
        y_dt_preds = DT.predict(X_test)
        y_rf_preds = RF.predict(X_test)
        recall_dt = recall_score(y_test[:, i], y_dt_preds[:, i])
        recall_rf = recall_score(y_test[:, i], y_rf_preds[:, i])
        dt_recalls.append(recall_dt)
        rf_recalls.append(recall_rf)
        f1_dt = f1_score(y_test[:, i], y_dt_preds[:, i])
        f1_rf = f1_score(y_test[:, i], y_rf_preds[:, i])
        f1_scores_dt.append(f1_dt)
        f1_scores_rf.append(f1_rf)

    print(dt_recalls)
    best_recall_dt = np.argmax(dt_recalls)
    print(f'best recall DT depth: {maxes_list[best_recall_dt]}')
    print(best_recall_dt)
    print(rf_recalls)
    best_recall_rf = np.argmax(rf_recalls)
    print(f'best recall RF depth: {maxes_list[best_recall_rf]}')
    print(best_recall_rf)
    print(f1_scores_dt)
    best_f1_dt = np.argmax(f1_scores_dt)
    print(best_f1_dt)
    print(f'best f1 DT depth: {maxes_list[best_f1_dt]}')
    print(f1_scores_rf)
    best_f1_rf = np.argmax(f1_scores_rf)
    print(best_f1_rf)
    print(f'best f1 RF depth: {maxes_list[best_f1_rf]}')
    types = ['recall_dt', 'recall_rf', 'f_1_dt', 'f_1_rf']
    values = [maxes_list[best_recall_dt], maxes_list[best_recall_rf],
              maxes_list[best_f1_dt], maxes_list[best_f1_rf]]
    plt.subplot(2,3,i+1)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)

    plt.bar(types, values)
plt.suptitle('failure 0 best hyperparameter')
plt.show()
#this analysis showed me that a max_depth and n_etimators of 32 is sufficient but for some failure types there are no predictions that were maide correctly.




SVM = SVC()
DT = DecisionTreeClassifier(max_depth=1000)
RF = RandomForestClassifier(max_depth=200)
MOCRF = MultiOutputClassifier(RF)
MOCDT = MultiOutputClassifier(DT)
MOCSVM = MultiOutputClassifier(SVM)
CCRF = ClassifierChain(RF)
CCDT = ClassifierChain(DT)
CCSVM = ClassifierChain(SVM)




#for (train_index, test_index) in folds:

