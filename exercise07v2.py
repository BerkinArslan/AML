import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt



from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle
class KNNClassifier():

    def distance(self, p):
        def calculate_distance(a, b):
            a = np.array(a)
            b = np.array(b)
            distance = np.sum(np.abs(a - b) ** p) ** (1/p)
            return distance
        return calculate_distance

    def __init__(self, k: int | float = 3,
                 metric: 'str | callable' = 'euclidean',
                 p: int = None):

        if isinstance(metric, str):
            if metric == 'euclidean':
                self.metric = self.distance(2)
            elif metric == 'minkowski':
                if p is None:
                    p = 2
                self.metric = self.distance(p)
            elif callable(metric):
                self.metric = metric
            else:
                raise ValueError('Invalid metric')

        if isinstance(k, int):
            self.k = k
            self.frac_k = None
        elif isinstance(k, float):
            self.frac_k = k
            self.k = None
        else:
            raise ValueError('Invalid k')

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def vote(self, index_array, y = None):
        if y is None:
            y = self.y
        votes = y[index_array]
        vote_nums = np.bincount(votes)
        majority_vote = np.argmax(vote_nums)
        return majority_vote

    def predict(self, X_query):
        all_closest = []
        for x in X_query:
            all_distances = [self.metric(x, point) for point in self.X]
            all_distances = np.array(all_distances)
            closest = np.argsort(all_distances)[: self.k]
            all_closest.append(closest)
        all_closest = np.array(all_closest)
        predictions = []
        for closest in all_closest:
            prediction = self.vote(closest)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def k_fold_split(x, y, k, seed = None):
    """
    Split the dataset into k equal sized folds
    :param x: data set (usually (N, n))
    :param y: data labels (usually (N, 1))
    :param k: number of folds
    :param seed: seed for random function
    :return:
    folded_x: np.ndarray of (k, N/k, n)
    folded_y: np.ndarray of (k, N/k)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    rng = np.random.default_rng()
    list_of_idx = np.array([i for i in range(x.shape[0])])
    shuffled = rng.permutation(list_of_idx)
    size_of_data = shuffled.shape[0]
    shuffled = list(shuffled)
    index_ranges = []
    i = 0
    for i in range(k):
        start = int(i * size_of_data / k)
        end = int((i + 1) * size_of_data / k)
        index_ranges.append([start, end])

    list_of_indices = []
    for i in range(len(index_ranges)):
        start = index_ranges[i][0]
        stop = index_ranges[i][1]
        #temp = shuffled[list_of_idx[start:stop]]
        temp = [shuffled[i] for i in list_of_idx[start:stop]]
        list_of_indices.append(temp)
    #print(list_of_indices)
    folded_x = []
    folded_y = []
    for index_spectrum in list_of_indices:
        #temp = [x[i] for i in index_spectrum]
        # folded_x.append(x[[x[i] for i in index_spectrum]])
        # folded_y.append(y[[y[i] for i in index_spectrum]])
        folded_x.append(x[index_spectrum])
        folded_y.append(y[index_spectrum])
    #folded_y = np.array(folded_y)
    #folded_x = np.array(folded_x)
    return folded_x, folded_y

def evaluate_cv(model: 'callable', fitter: 'callable', predictor: 'callable',
                x: np.ndarray, y, k: int = 5, seed = None, **kwargs ):
    """
    will run k times k-fold
    :param model: callable, model to run the data on
    :param x: data
    :param y: labels
    :return: mean accuracy
    """
    folded_x, folded_y = k_fold_split(x, y, k)#(k, N/k, n) , (k, N/k)
    accuracies = []
    model_names = []
    for i in range(k):
        model_name = f'Model {i}'
        model_names.append(model_name)
        copy_folded_x = folded_x.copy()
        copy_folded_y = folded_y.copy()
        test_split_x = copy_folded_x[i]
        test_split_y = copy_folded_y[i]
        train_x = copy_folded_x[:i] + copy_folded_x[i + 1:]
        train_y = copy_folded_y[:i] + copy_folded_y[i + 1:]
        # train_x = np.delete(copy_folded_x, i, axis = 0)
        # train_y = np.delete(copy_folded_y, i, axis = 0)
        # train_x = train_x.reshape(-1, x.shape[1])
        # train_y = train_y.reshape(-1)
        train_x = np.vstack(train_x)
        train_y = np.hstack(train_y)
        ml_model =  model(**kwargs)
        fitted_model = fitter(ml_model, train_x, train_y)
        y_pred = predictor(fitted_model, test_split_x)
        y_true = test_split_y
        accuracy_of_model = accuracy(y_true, y_pred)
        accuracies.append(accuracy_of_model)
        total_acccuracy = sum(accuracies) / len(accuracies)
    plt.bar(model_names, accuracies)
    min_score = min(accuracies)
    max_score = max(accuracies)
    margin = (max_score - min_score) / 5
    plt.title(f'the model with the keyword argument: {kwargs}')
    plt.ylim(min_score - margin, max_score + margin)
    plt.show()
    return total_acccuracy





if __name__ == '__main__':
    #data = np.genfromtxt("water_potability.csv", delimiter = ',')
    #print(data)
    #dtc = DecisionTreeClassifier()
    #dtc.fit()
    df_water_data = pd.read_csv('water_potability.csv')


    X = df_water_data.iloc[:, : -1]
    X = X.to_numpy()

    y = df_water_data.iloc[:, -1]
    y = y.to_numpy()
    y = y[~np.isnan(X).any(axis = 1)]
    X = X[~np.isnan(X).any(axis = 1)]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    accuracy_of_rfc = evaluate_cv(RandomForestClassifier, fitter=lambda model, x, y: model.fit(x, y),
                predictor=lambda model, x: model.predict(x), x = X, y = y)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print(f'K-Fold accuracy of RandomForestClassifier: {accuracy_of_rfc} \n')

    X, y = shuffle(X, y, random_state=42)
    X_splits = KFold(5).split(X)
    dtc = DecisionTreeClassifier()

    results = {
        'DT': [],
        'RF': [],
        'CKNN': [],
        'KNN': [],
        'GBC': []
    }


    for i, split in enumerate(X_splits):
        train_idx = split[0]
        test_idx = split[1]
        dtc.fit(X[train_idx], y[train_idx])
        y_pred = dtc.predict(X[test_idx])
        y_true = y[test_idx]
        # print('DT Classifier: ')
        # print(accuracy(y_true, y_pred))
        # print('\n')
        results['DT'].append(accuracy(y_true, y_pred))

        train_X = X[train_idx]
        train_y = y[train_idx]
        test_X = X[test_idx]
        test_y = y[test_idx]

        rfc = RandomForestClassifier(class_weight= 'balanced').fit(train_X, train_y)
        y_pred = rfc.predict(test_X)
        # print('RF Classifier: ')
        # print(accuracy(y_true, y_pred))
        # print('\n')
        results['RF'].append(accuracy(y_true, y_pred))


        knn_custom = KNNClassifier(3).fit(train_X, train_y)
        # y_pred = knn_custom.predict(test_X)
        # print('Custom KNN Classifier: ')
        # print(accuracy(y_true, y_pred))
        # print('\n')
        results['CKNN'].append(accuracy(y_true, y_pred))

        knn = KNeighborsClassifier(5).fit(train_X, train_y)
        y_pred = knn.predict(test_X)
        # print('KNN Classifier: ')
        # print(accuracy(y_true, y_pred))
        # print('\n')
        results['KNN'].append(accuracy(y_true, y_pred))


        gbc = GradientBoostingClassifier().fit(train_X, train_y)
        y_pred = gbc.predict(test_X)
        #print('Gradient Boosting Classifier: ')
        #print(accuracy(y_true, y_pred))
        #print('\n')
        results['GBC'].append(accuracy(y_true, y_pred))

    df_results = pd.DataFrame(results)
    print(df_results)
    best_hyperparameters = []
    mean_total_acc = []
    names = []
    accs = []
    best_acc = []
    list_ = [1, 3, 5, 7, 11]
    for i in [1, 3, 5, 7, 11]:
        #knn = KNeighborsClassifier(n_neighbors= k)

        acc = evaluate_cv(KNeighborsClassifier, lambda model, x, y: model.fit(x, y),
                    predictor = lambda model, x: model.predict(x), x = X, y = y, k = 5, n_neighbors = i)
        accs.append(acc)
    max_index = accs.index(max(accs))
    max_accuracy = accs[max_index]
    best_acc.append(max_accuracy)
    max_hyper_parameter = list_[max_index]
    best_hyperparameters.append(max_hyper_parameter)
    mean_acc = sum(accs) / len(accs)
    mean_total_acc.append(mean_acc)
    accs = []
    names.append('KNeighborsClassifier')
    list_ = [3, 4, 5, 6, 10, 25]
    for i in [3, 4, 5, 6, 10, 25]:
        acc = evaluate_cv(DecisionTreeClassifier, lambda model, x, y: model.fit(x, y),
                    predictor=lambda model, x: model.predict(x), x=X, y=y, k=5, max_depth=i)
        accs.append(acc)
    max_index = accs.index(max(accs))
    max_accuracy = accs[max_index]
    best_acc.append(max_accuracy)
    max_hyper_parameter = list_[max_index]
    best_hyperparameters.append(max_hyper_parameter)
    mean_acc = sum(accs) / len(accs)
    mean_total_acc.append(mean_acc)
    accs = []
    names.append('DecisionTreeClassifier')
    list_ = [20, 50, 100, 200]
    for i in [20, 50, 100, 200]:
        acc = evaluate_cv(RandomForestClassifier, lambda model, x, y: model.fit(x, y),
                    predictor=lambda model, x: model.predict(x), x=X, y=y, k=5, n_estimators=i)
        accs.append(acc)
    max_index = accs.index(max(accs))
    max_hyper_parameter = list_[max_index]
    best_hyperparameters.append(max_hyper_parameter)
    max_accuracy = accs[max_index]
    best_acc.append(max_accuracy)
    mean_acc = sum(accs) / len(accs)
    mean_total_acc.append(mean_acc)
    accs = []
    names.append('RandomForestClassifier')
    for i in [10, 20, 50, 100]:
        acc = evaluate_cv(AdaBoostClassifier, lambda model, x, y: model.fit(x, y),
                    predictor=lambda model, x: model.predict(x), x=X, y=y, k=5, estimator= DecisionTreeClassifier(max_depth=100),  n_estimators=i)
        accs.append(acc)
    max_index = accs.index(max(accs))
    max_hyper_parameter = list_[max_index]
    best_hyperparameters.append(max_hyper_parameter)
    max_accuracy = accs[max_index]
    best_acc.append(max_accuracy)
    mean_acc = sum(accs) / len(accs)
    mean_total_acc.append(mean_acc)
    accs = []
    names.append('AdaBoost')

    df_choose_best_hyperparameter_dict = {
        'Name of the model': [],
        'Best Hyperparameter': [],
        'Accuracy': []
    }

    df_choose_best_hyperparameter_dict['Name of the model'] = names
    df_choose_best_hyperparameter_dict['Best Hyperparameter'] = best_hyperparameters
    df_choose_best_hyperparameter_dict['Accuracy'] = best_acc

    df_choose_best_hyperparameter = pd.DataFrame(df_choose_best_hyperparameter_dict)
    print(df_choose_best_hyperparameter)
    plt.bar(names, mean_total_acc)
    max_score = max(mean_total_acc)
    min_score = min(mean_total_acc)
    margin = (max_score - min_score) / 5
    plt.ylim(min_score - margin, max_score + margin)
    plt.show()
        ##################################################################################






