import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from exercise07v2 import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
def binarize_predictions(y_pred: np.ndarray|list, threshold: float = 0.5):
    output = []
    for prediction in y_pred:
        if prediction < threshold:
            output.append(0)
        else:
            output.append(1)
    output = np.array(output)
    return output

def compute_precision_recall_curve(y_pred_proba: np.ndarray,
                                   y_true:np. ndarray, steps: int = 100):

    precisions = []
    recalls = []
    thresholds = []
    for i in range(steps):
        threshold = i/steps
        y_output = binarize_predictions(y_pred_proba,
                                        threshold=threshold)

        precision = precision_score(y_true, y_output, zero_division=0)
        recall = recall_score(y_true, y_output, zero_division=0)
        if precision + recall > 0:
            thresholds.append(threshold)

            precisions.append(precision)
            recalls.append(recall)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    thresholds = np.array(thresholds)

    return precisions, recalls, thresholds

def calculate_f_n_score(precision: float | np.ndarray,
                        recall: float | np.ndarray, n: int = 1):
    f_n_scores = ((n ** 2 + 1) * precision * recall) / (recall + (n ** 2 * precision))
    return f_n_scores







if __name__ == '__main__':
    #load clean and prepare the data
    df_water_potability = pd.read_csv('water_potability.csv')
    df_water_potability_clean = df_water_potability[~df_water_potability.isna().any(axis = 1)]
    water_potability_clean = np.array(df_water_potability_clean)
    X = water_potability_clean[:, :-1]
    y = water_potability_clean[:, -1]

    X_scaled = StandardScaler().fit_transform(X)
    n = X.shape[0]
    indices = [i for i in range(n)]
    #np.random.seed(468)
    indices = np.random.permutation(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled)

    ##building the classifier
    # classifier = RandomForestClassifier(n_estimators = 500).fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    # disp.plot()
    # plt.show()
    # print(cm)
    # recall = recall_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    # f_1 = f1_score(y_test, y_pred)
    # print(f'Recall: {recall}')
    # print(f'precision: {precision}')
    # print(f'f1: {f_1}')
    #
    # #try again
    # classifier = RandomForestClassifier(n_estimators=100)
    # classifier.fit(X_train, y_train)
    # y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    # precisions, recalls, thresholds = compute_precision_recall_curve(y_pred_proba, y_test)
    # plt.plot(recalls, precisions, marker = 'o')
    # for i in range(0, len(thresholds), 5):
    #     plt.annotate(f'{thresholds[i]:.2f}',
    #                  (recalls[i], precisions[i]),
    #                  textcoords="offset points", xytext=(0, 10),
    #                  ha='center',
    #                  fontsize = 8,
    #                  rotation = 45)
    # plt.grid()
    # plt.title('Precision-Recall-Curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend()
    # plt.show()
    #
    # display = PrecisionRecallDisplay(precisions, recalls,
    #                                  estimator_name= 'RandomForestEstimator',
    #                                  pos_label='potable water',
    #                                  )
    # display.plot()
    # plt.show()
    #
    # regressor = LogisticRegression()
    # regressor.fit(X_train, y_train)
    # y_pred_proba = regressor.predict_proba(X_test)[:, 1]
    # precisions, recalls, thresholds = compute_precision_recall_curve(y_pred_proba, y_test)
    # display = PrecisionRecallDisplay(precisions, recalls,
    #                                  estimator_name='Logistic Regressor',
    #                                  pos_label='potable water',
    #                                  )
    # display.plot()

    # plt.show()

    classifiers = {
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100),
        'K Nearest Neighbours': KNeighborsClassifier(n_neighbors=11),
        'AdaBoost': AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth=10))
    }

    for name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = compute_precision_recall_curve(y_pred_proba, y_test, steps=50)
        plt.plot(recalls, precisions, marker = 'o', label = name)
        optimal_idx = np.argmax(precisions)
        optimal_threshold = thresholds[optimal_idx]
        for i in range(0, len(thresholds), 5):
            plt.annotate(f'{thresholds[i]:.2f}',
                         (recalls[i], precisions[i]),
                         textcoords="offset points", xytext=(0, 10),
                         ha='center',
                         fontsize=8,
                         rotation=45)
        print(f'{name} optimal threshold: {optimal_threshold:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.show()


    #this is actually good and makes sense
    #since we dont want to risk peoples lives
    #we are very careful about labeling as potable
    optimal_threshold = 0.75
    optimal_classifier = RandomForestClassifier(n_estimators=100)
    optimal_classifier.fit(X_train, y_train)
    y_pred_proba = optimal_classifier.predict_proba(X_test)[:, 1]
    y_pred = binarize_predictions(y_pred_proba, threshold=optimal_threshold)
    confusion_matrix_ = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix_)
    display.plot()
    plt.show()


    for name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = compute_precision_recall_curve(y_pred_proba, y_test, steps=50)
        f_1_score = calculate_f_n_score(precisions, recalls)
        optimal_idx = np.argmax(f_1_score)
        optimal_threshold = thresholds[optimal_idx]
        print(f'{name} optimal threshold: {optimal_threshold:.2f}')
        plt.plot(thresholds, f_1_score, label=name)
    plt.xlabel('Decision Threshold')
    plt.ylabel('F1 Score')
    plt.grid()
    plt.legend()
    plt.savefig('f1_score_vs_threshold.png')
    plt.show()


    #for water potability this optimaized parameter does not make much sense:
    #because this least to a lot of non-potable water being labeled as potable
    #which leads to health risks
    #but for education purpouses:
    optimal_threshold = 0.32
    optimal_classifier = RandomForestClassifier(n_estimators=100)
    optimal_classifier.fit(X_train, y_train)
    y_pred_proba = optimal_classifier.predict_proba(X_test)[:, 1]
    y_pred = binarize_predictions(y_pred_proba, threshold=optimal_threshold)
    confusion_matrix = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix)
    display.plot()
    plt.show()







