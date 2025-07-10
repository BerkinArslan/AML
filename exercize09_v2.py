import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve
from sklearn.metrics import classification_report


df_raw = pd.read_csv("ai4i2020.csv")
pd.set_option('display.max_columns', None)

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

def calculate_weights(vector):
    n_0 = np.sum(vector == 0)
    n_1 = np.sum(vector == 1)
    n_total = vector.shape[0]
    weight_0 = 1
    weight_1 = (n_0/n_total) * (n_total/n_1)
    weights = []
    for status in vector:
        if status == 0:
            weights.append(weight_0)
        elif status == 1:
            weights.append(weight_1)

    return weights


def find_optimal_hyperparameter(model: callable, keyword: str,
                                list_of_hyperparameter: np.ndarray,
                                X_train: np.ndarray,
                                X_test: np.ndarray,
                                y_train: np.ndarray,
                                y_test: np.ndarray):
    recalls = []
    f1s = []
    for i in range(list_of_hyperparameter.shape[0]):
        param = {keyword: list_of_hyperparameter[i],
                 'class_weight': 'balanced'}
        #predictor_ = MultiOutputClassifier(model(**param))
        predictor_ = ClassifierChain(model(**param))

        #predictor_ = model(keyword=list_of_hyperparameter[i])
        predictor_.fit(X_train, y_train)
        y_pred = predictor_.predict(X_test)

        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        recalls.append(recall)
        f1s.append((f1))

    figure = plt.figure(figsize=(15, 8))
    plt.subplot(1,2,1)
    plt.bar(range(len(list_of_hyperparameter)), recalls)
    plt.xticks(range(len(list_of_hyperparameter)), list_of_hyperparameter)
    plt.title('Recall')
    plt.subplot(1,2,2)
    plt.bar(range(len(list_of_hyperparameter)), f1s)
    plt.xticks(range(len(list_of_hyperparameter)), list_of_hyperparameter)
    plt.title('F1 Scores')
    plt.show()

    keys = {
        'Hyperparameters': list_of_hyperparameter,
        'Recalls': recalls,
        'F1_scores': f1s
    }

    df_scores = pd.DataFrame(keys)
    return df_scores





if __name__ == '__main__':
    df_raw = pd.read_csv("ai4i2020.csv")
    pd.set_option('display.max_columns', None)
    #print(df_raw)
    y = df_raw.iloc[:, -6:]
    y = np.array(y)
    X = df_raw.iloc[:, 2:-6]
    X = np.array(X)
    for i, type in enumerate(X[:, 0]):
        if type == 'L':
            X[i, 0] = 0
        elif type == 'M':
            X[i, 0] = 1
        elif type == 'H':
            X[i, 0] = 2
    calculate_stats_columns(df_raw)
    weights_machine_failure = calculate_weights(df_raw['Machine failure'])
    list_of_hp = np.array([2,4,8,16,32,64,128,256])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y[:, 0])
    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    # find_optimal_hyperparameter(RandomForestClassifier, 'n_estimators', list_of_hp,
    #                             X_train, X_test, y_train, y_test)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # SVM = SVC(class_weight='balanced')
    # model = MultiOutputClassifier(SVM)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # recall = recall_score(y_test, y_pred, average = 'macro')
    # f1 = f1_score(y_test, y_pred, average= 'macro')
    # print(f'recall score of the SVM: {recall}')
    # print(f'F1 Score of SVM: {f1}')
    # print(classification_report(y_test, y_pred, zero_division=0))
    # # cm = confusion_matrix(y_test, y_pred)
    # # disp = ConfusionMatrixDisplay(cm)
    # # disp.plot()
    # # plt.show()
    #
    #
    #
    # SVM = SVC(class_weight='balanced')
    # model = ClassifierChain(SVM)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # recall = recall_score(y_test, y_pred, average='macro')
    # f1 = f1_score(y_test, y_pred, average='macro')
    # print(f'recall score of the SVM balanced: {recall}')
    # print(f'F1 Score of SVM balanced: {f1}')
    # print(classification_report(y_test, y_pred, zero_division=0))
    #
    # for i in range(6):
    #     cm = confusion_matrix(y_test[:, i], y_pred[:, i])
    #     disp = ConfusionMatrixDisplay(cm)
    #     disp.plot()
    #     plt.title(f'SVM for the {i}th property')
    #     plt.show()
    #
    #
    # SVM = SVC(class_weight='balanced')
    # model = SVM
    # model.fit(X_train, y_train[:, 0])
    # y_pred = model.predict(X_test)
    # recall = recall_score(y_test[:, 0], y_pred)
    # f1 = f1_score(y_test[:, 0], y_pred)
    # print(f'recall score of the SVM: {recall}')
    # print(f'F1 Score of SVM: {f1}')
    # cm = confusion_matrix(y_test[:, 0], y_pred)
    # disp = ConfusionMatrixDisplay(cm)
    # disp.plot()
    # plt.title('SVM')
    # plt.show()
    #
    # RF = RandomForestClassifier(n_estimators=128, class_weight='balanced')
    # model_ = ClassifierChain(RF)
    # model_.fit(X_train, y_train)
    # y_pred = model_.predict(X_test)
    # recall = recall_score(y_test, y_pred, average='macro')
    # f1 = f1_score(y_test, y_pred, average='macro')
    # print(f'recall score of the RFCC: {recall}')
    # print(f'F1 Score of RFCC: {f1}')
    # print(classification_report(y_test, y_pred, zero_division=0))
    #
    # model_ = RandomForestClassifier(n_estimators=128, class_weight='balanced')
    # model_.fit(X_train, y_train[:, 0])
    # y_pred = model_.predict(X_test)
    # recall = recall_score(y_test[:, 0], y_pred)
    # f1 = f1_score(y_test[:, 0], y_pred)
    # print(f'recall score of the RF: {recall}')
    # print(f'F1 Score of RF: {f1}')
    # print(classification_report(y_test[:, 0], y_pred, zero_division=0))
    # cm = confusion_matrix(y_test[:, 0], y_pred)
    # disp = ConfusionMatrixDisplay(cm)
    # disp.plot()
    # plt.title('RF')
    # plt.show()
    #
    # RF = RandomForestClassifier(n_estimators=128, class_weight='balanced')
    # model_ = MultiOutputClassifier(RF)
    # model_.fit(X_train, y_train)
    # y_pred = model_.predict(X_test)
    # recall = recall_score(y_test, y_pred, average='macro')
    # f1 = f1_score(y_test, y_pred, average='macro')
    # print(f'recall score of the RFMO: {recall}')
    # print(f'F1 Score of RFMO: {f1}')
    # print(classification_report(y_test, y_pred, zero_division=0))



    """
    There seems to be a trade off here. SVM predicts too much 1 but also cathches all of the problems.
    RF goes through without picking up nything. 
    lets try a probabilistic approach with ever getting higher decision threshold for RF
    """

    model_ = RandomForestClassifier(n_estimators=128, class_weight='balanced')
    model_.fit(X_train, y_train[:, 0])
    y_pred = model_.predict_proba(X_test)
    thresholds = np.arange(0.05, 0.51, 0.05).tolist()
    for threshold in thresholds:
        y_pred_ = (y_pred[:, 1] >= threshold).astype(int)
        recall = recall_score(y_test[:, 0], y_pred_)
        f1 = f1_score(y_test[:, 0], y_pred_)
        print(f'recall score of the RF with threshold {threshold}: {recall}')
        print(f'F1 Score of RF with threshold {threshold}: {f1}')
        print(classification_report(y_test[:, 0], y_pred_, zero_division=0))
        cm = confusion_matrix(y_test[:, 0], y_pred_)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title(f'RF with threshold {threshold}')
        plt.show()

    """
    A RF with prebability calcualtion can have a better recall score. 
    it is ok to predict as faulty most of the times but not ok to predict ok whe it is not
    the new trade off is how much precision and recall do we want?
    """
#
    # thresholds = np.arange(0, 1.01, 0.05).tolist()
    # for threshold in thresholds:
    #     RF = RandomForestClassifier(n_estimators=128, class_weight='balanced')
    #     model_ = MultiOutputClassifier(RF)
    #     model_.fit(X_train, y_train)
    #     y_pred_proba = model_.predict_proba(X_test)
    #     y_pred = np.array([proba[:, 1] for proba in y_pred_proba]).T
    #     y_pred = (y_pred >= threshold).astype(int)
    #     recall = recall_score(y_test, y_pred, average='macro')
    #     f1 = f1_score(y_test, y_pred, average='macro')
    #     print(f'recall score of the RFMO: {recall}')
    #     print(f'F1 Score of RFMO: {f1}')
    #     print(classification_report(y_test, y_pred, zero_division=0))


    for i in range(6):
        plt.subplot(3, 2, i + 1)
        RF = RandomForestClassifier(n_estimators=128, class_weight='balanced')
        model_ = MultiOutputClassifier(RF)
        model_.fit(X_train, y_train)
        y_pred_proba = model_.predict_proba(X_test)
        y_scores = np.array([proba[:, 1] for proba in y_pred_proba]).T
        pr, rc, thr = precision_recall_curve(y_test[:, i], y_scores[:, i])
        plt.plot(rc, pr)
        plt.title(f'prc of the {i}th failure', fontsize=5)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        plt.xlabel('recall', fontsize = 5)
        plt.ylabel('precision', fontsize = 5)
        plt.suptitle('MultiOutputClassifier with RandomForests')
    plt.tight_layout()
    plt.show()

    for i in range(6):
        plt.subplot(3, 2, i + 1)
        RF = RandomForestClassifier(n_estimators=128, class_weight='balanced')
        model_ = ClassifierChain(RF)
        model_.fit(X_train, y_train)
        y_pred_proba = model_.predict_proba(X_test)
        y_scores = np.array([proba for proba in y_pred_proba])
        pr, rc, thr = precision_recall_curve(y_test[:, i], y_scores[:, i])
        plt.plot(rc, pr)
        plt.title(f'prc of the {i}th failure', fontsize=5)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        plt.xlabel('recall', fontsize = 5)
        plt.ylabel('precision', fontsize = 5)
        plt.suptitle('ClassifierChain with RandomForests')
    plt.tight_layout()
    plt.show()

    """
    From my analysis I can see that MultiOutputClassifier is better, so we are going to use MultiOutputClassifier.
    
    1.  Model preforms somewhat good for general failure
    2. The model predicts 2rd, 3th, 4th failure types really good
    3. The model cannot predict the 1st and 5th failure type
    
    To improve the model I suggest:
    
    1. Taking out the Data with th e 1st and 5th type of failure for the model.
    2. first predicting the Machine Failure as whole
    3. from all the failure data then we predict whcih type of failure that is
    4. for this we should not use multi label classification because this is not multi label
    5. I can use RF but whit more classes and not one hot encoded.
    6. I will one hot encode the classes afterwards
    """









