import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score, f1_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.metrics import average_precision_score, ConfusionMatrixDisplay
# Intereger Encoding
pd.set_option('display.max_columns', 500)
df_machine_report = pd.read_csv('ai4i2020.csv')
df_machine_report = shuffle(df_machine_report)



""" X:
There are 3 classes for the product type (nominal data):
Needs to be OneHotEncoded® :)

Product ID and UDI is not important. they will not be taken in to considiration

The rest of the values are rational no operation needed
"""

""" Y:
There is one superior label that all the other labels are dependent on
then there is multi class labels

Machine Failure is the superior class and all the other labels are in y are
part of a multi class label: which type of machine failure?
"""

############

"""
STRATEGY:

2 step classification:

1. Binary classifier for machine failure
-> random forests/SVM/Logistic Regression

2. Multi class classification
take only the ones where machien failure is seen
then:
-> Random Forest: Decode from OHE 
RF can hadle multi class classification
-> ANN can handle OHE multi class good with softmax activation in the end layer

"""

##############

x = df_machine_report.iloc[:, 3:8]
x_machine_type = df_machine_report.iloc[:, 2]
x_ohe_machine_type = {
    'M': x_machine_type == 'M',
    'L': x_machine_type == 'L',
    'H': x_machine_type == 'H',
}
x_ohe_machine_type = pd.DataFrame(x_ohe_machine_type)
x_ohe_machine_type = x_ohe_machine_type.astype(int)

x = pd.concat([x_ohe_machine_type, x], axis=1)
#print(x)
scaler = StandardScaler()
x = scaler.fit_transform(x)
#print(x)
# print(x.head())

# print(x.head())
y = df_machine_report.iloc[:, -6:]
# print(y.head())

# KF = StratifiedKFold().split(x, y)
# print(KF)

y_failure_sup = y.iloc[:, 0]
#print(y_failure_sup)


# n_trees = [32,64,128,256]
# i = 1
#
# names = ['PRC n_estimator = 32', 'PRC n_estimator = 64', 'PRC n_estimator = 128', 'PRC n_estimator = 256']
# aps = []
# f1s = []
# best_thresholds = []
# for n_tree in n_trees:
#     plt.subplot(2, 2, i)
#     plt.title(names[i - 1])
#     i = i + 1
#     ap_temp = []
#     kf_superior = StratifiedKFold().split(x, y_failure_sup)
#     for fold, (train_idx, test_idx) in enumerate(kf_superior):
#         x_train, x_test = x[train_idx], x[test_idx]
#         y_train, y_test = y_failure_sup.iloc[train_idx], y_failure_sup.iloc[test_idx]
#         RF = RandomForestClassifier(n_estimators=n_tree).fit(x_train, y_train)
#         y_pred_proba = RF.predict_proba(x_test)[:, 1]
#         precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
#         ap = average_precision_score(y_test, y_pred_proba)
#         ap_temp.append(ap)
#         f1 = 2 * precision[1:] * recall[1:] / (precision[1:] + recall[1:] + 1e-8)
#         best_f1_idx = np.argmax(f1)
#         f1s.append(f1[best_f1_idx])
#         best_thresholds.append(thresholds[best_f1_idx])
#         plt.plot(recall, precision, label='Precision-Recall curve')
#     ap_mean = np.mean(ap_temp)
#     aps.append(ap_mean)
# plt.show()
# print(aps)
# print(f'RF best thresholds: {best_thresholds}')
# print(f'RF best f1s: {f1s}')
"""
results were bad:
forgot to normalize and mix...
NEVER FORGET TO NORMALIZE DATA
"""

"""
Random Forest with 128 estimators is enough!
lets try the same thing with SVM
"""
# aps = []
# names = []
# f1s = []
# kf_superior = StratifiedKFold().split(x, y_failure_sup)
# for fold, (train_idx, test_idx) in enumerate(kf_superior):
#     x_train, x_test = x[train_idx], x[test_idx]
#     y_train = y_failure_sup.iloc[train_idx]
#     y_test = y_failure_sup.iloc[test_idx]
#     svm = SVC(class_weight='balanced', kernel='rbf')
#     svm.fit(x_train, y_train)
#     y_pred = svm.predict(x_test)
#     recall = recall_score(y_test, y_pred)
#     aps.append(recall)
#     f1 = f1_score(y_test, y_pred)
#     f1s.append(f1)
#     names.append(f'Fold {fold}')
#     cm = confusion_matrix(y_test, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot()
#     plt.show()
# print(aps)
# print(f'SVM f1 scores: {f1s}')
# #plt.bar(names, aps)
# #plt.show()

"""
balanced weighted SVM performs so much better than SVM,
I already used stratified kfold but looks like there is still imbalance....
Thats why I will run with weighet random forests as well...
Somehow the scores fore balanced wight trees got even worse...
I will continue to inverstigate SVM
polynomial nad rbf is giving the same values... lets see prc
"""

# aps = []
# names = []
# f1s = []
# best_thresholds = []
# kf_superior = StratifiedKFold().split(x, y_failure_sup)
# for fold, (train_idx, test_idx) in enumerate(kf_superior):
#     plt.subplot(2, 3, fold + 1)
#     x_train, x_test = x[train_idx], x[test_idx]
#     y_train = y_failure_sup.iloc[train_idx]
#     y_test = y_failure_sup.iloc[test_idx]
#     svm = SVC(class_weight='balanced', kernel='rbf', probability=True)
#     svm.fit(x_train, y_train)
#     y_pred_proba = svm.predict_proba(x_test)[:, 1]
#     precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
#     f1 = 2 * precision[1:] * recall[1:] / (precision[1:] + recall[1:] + 1e-8)
#     best_f1_idx = np.argmax(f1)
#     f1s.append(f1[best_f1_idx])
#     best_thresholds.append(thresholds[best_f1_idx])
#     ap = average_precision_score(y_test, y_pred_proba)
#     aps.append(ap)
#     plt.plot(recall, precision, label='Precision-Recall curve')
# print(aps)
# print(best_thresholds)
# print(f'SVM f1s {f1s}')
# plt.show()

"""
my investigation revealed that random forest is a better classifier because allows fore better precision with same recall.
RANDOM FOREST WITH 128 ESTIMATORS WILL BE USED FROM NOW ON FOR SUPERIOR CLASS
"""


"""
what to do now:
1. used RF n_estimators 128 for classification of error.
2. mask the failures and teach for multi class classification.
3. find out the best ML tool for that.
4. if enough time try for ANN as well.
"""

sup_model = RandomForestClassifier(n_estimators=128)
x_train, x_test, y_train, y_test = train_test_split(x, y_failure_sup, test_size=0.2, stratify=y_failure_sup)
sup_model.fit(x_train, y_train)
y_pred_proba = sup_model.predict_proba(x_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=[12, 8])
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
for i in range(thresholds.shape[0] - 1):
    pre = precision[i]
    rec = recall[i]
    plt.scatter(rec, pre)
    plt.text(rec, pre, f'{thresholds[i]:.2f}', fontsize = 8)
plt.show()

threshold = 0.05
"""
depending on maintance cost this value could be even lower
imagine: maintanence costs 10euros
missed maintance costs 10000 euros
then precision under 0.05 still makes sense
"""
y_pred = y_pred_proba > threshold
y_pred = y_pred.astype(int)
recall_ = recall_score(y_test, y_pred)
precision_ = precision_score(y_test, y_pred)
print(f'precision: {precision_:.2f}')
print(f'recall: {recall_:.2f}')
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

"""
STAGE 2: GUESS WHAT TYPE OF ERROR
for this chapter we can use1:
ANN or RF
to use RF we need to integer code the problem
for an ANN OHE is good
"""

df_machine_report_only_failures = df_machine_report[df_machine_report['Machine failure'] == 1]
#print(df_machine_report_only_failures.head())

x = df_machine_report_only_failures.iloc[:, 3:8]
x_machine_type = df_machine_report_only_failures.iloc[:, 2]
x_ohe_machine_type = {
    'M': x_machine_type == 'M',
    'L': x_machine_type == 'L',
    'H': x_machine_type == 'H',
}
x_ohe_machine_type = pd.DataFrame(x_ohe_machine_type)
x_ohe_machine_type = x_ohe_machine_type.astype(int)
x = pd.concat([x_ohe_machine_type, x], axis=1)
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)


y_types = df_machine_report_only_failures.iloc[:, -5:]
y_ann = y_types




failure_types = []
valid_indices = []
for i in range(y_types.shape[0]):
    if all(y_types.iloc[i] == 0):
        continue  # skip rows with no failure
    valid_indices.append(i)
    for j in range(y_types.shape[1]):
        if y_types.iloc[i, j] == 1:
            failure_types.append(j)
            break
x = x[valid_indices]

#y_types = pd.get_dummies(failure_types).astype(int)

print(y_types)
print(x)

y_types= {
    'Failure type': failure_types
}
y_types = pd.DataFrame(y_types)

x_train, x_test, y_train, y_test = train_test_split(x, y_types, test_size=0.2, stratify=y_types)
plt.figure(figsize=[12, 8])
#lets try RandomForests:
n_estimators = [16, 32, 64, 128, 256, 512]
aps = []
for n in n_estimators:
    RF_model = RandomForestClassifier(n_estimators=n)
    RF_model.fit(x_train, y_train)
    #y_pred_proba = RF_model.predict_proba(x_test)[:, 1]
    y_pred = RF_model.predict(x_test)
    print(classification_report(y_test, y_pred, digits=3))



"""
Looks like for this a RandomForestClassifier with n_estimators=64 is enough for this task
"""

# CHATGPT IMPLEMENTATION FOR ANN
y_ohe = y_ann.iloc[valid_indices]
# One-hot encode labels



# Train-test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, test_size=0.2, random_state=42)

# Build ANN with Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Input(shape=(x.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_ohe.shape[1], activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=32, verbose=1)

y_pred_proba = model.predict(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)
# Convert one-hot encoded y_test to class indices
y_true = np.argmax(y_test.values, axis=1)

# Now compare predictions to true labels
accuracy = np.mean(y_pred == y_true)
print(f'Accuracy: {accuracy:.3f}')

"""
ANN is an even better fit
"""








