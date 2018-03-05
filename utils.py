import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import pdist
from sklearn.model_selection import PredefinedSplit

def training_data():
    # X training set
    X_train = np.load("data/X_train.npy")

    # X test set
    X_test = np.load("data/X_test.npy")

    # Y training set
    y_train_csv = np.array(pd.read_csv("data/y_train.csv"))
    # Create an index of class names
    # ---------------------------------------------------------------------
    
    # Label encoder for Y scene_label column
    y_label_encoder = preprocessing.LabelEncoder()
    y_label_encoder.fit(y_train_csv[0:,1])
    y_train = y_label_encoder.transform(y_train_csv[0:, 1])

    return X_train, y_train, X_test, y_label_encoder

def extract_feature(train, test, method="B", scale=False):
    # Extract features
    # ---------------------------------------------------------------------
    X_train = []
    X_test = []
    if (method == "A"):
        # A:
        # 20040-dimensional vector from each sample
        X_train = train.reshape(-1, train.shape[1] * train.shape[2])
        X_test =  test.reshape(-1, train.shape[1] * train.shape[2])
    
    elif(method == "B"):
        # B:
        # (4500,40) data matrix, each of the 4500 samples is a vector of 40 frequency bins averaged over time
        X_train = np.mean(train,axis=2)
        X_test = np.mean(test,axis=2)
    elif(method == "C"):
        # C:
        # (4500,501) data matrix, each of the 4500 samples is a vector of 501 time points averaged frequencies
        X_train = np.mean(train,axis=1)
        X_test = np.mean(test,axis=1)

    elif method == "D":
        # D:
        # (4500,780) data matrix, each of the 4500 samples is a vectorized distance matrix of the cosine/correlation/euclidean/seuclidean... 
        # distances between time point vectors on each frequency bin
        for index in range(np.shape(train)[0]):
            observations = train[index,:,:]
            dist_train = pdist(observations, 'seuclidean')
            X_train.append(dist_train)

        for index in range(np.shape(test)[0]):
            observations = test[index,:,:]
            dist_test = pdist(observations, 'seuclidean')
            X_test.append(dist_test)

        X_train = np.array(X_train)
        X_test = np.array(X_test)
    elif method == "plain":
        X_train = train
        X_test = test

    # Scale
    if scale:
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test


def write_submission(name, y):
    filename = "".join(["submission_", name, ".csv"])
    with open(filename, "w") as fp:
        fp.write("Id,Scene_label\n")
        for i, label in enumerate(y):
            fp.write("%d,%s\n" % (i, label))


def report(results, name, n_top=3):
    print(name,"Grid scores on development set:")
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print(name, "Parameters: {0}".format(results['params'][candidate]))
            print("")

def split_data(X_train, y_train):
    y_cv_split = np.array(pd.read_csv("data/crossvalidation_train.csv"))
    train_indices = y_cv_split[y_cv_split[:,2] == "train"][:,0].astype(int)
    test_indices = y_cv_split[y_cv_split[:,2] == "test"][:,0].astype(int)

    X_test = X_train[train_indices, :]
    X_cv = X_train[test_indices, :]

    y_test = y_train[train_indices]
    y_cv = y_train[test_indices]

    # Array which contains "test" values marked as True and "train" as false
    mask = y_cv_split[:, 2]  == "test"
    mask = mask.astype(int)
    # "test" values marked as 0 and train values "-1" 
    mask -= 1
    cv_split_indices = PredefinedSplit(mask).split()

    return X_test, X_cv, y_test, y_cv, cv_split_indices
    

def test_classifier(Clf, param_grid, feature_extract):
    X_train, y_train, X_test, y_label_encoder = training_data()
    X_train, X_test = extract_feature(X_train, X_test, feature_extract)
    X_t, X_cv, y_t, y_cv, cv_indices = split_data(X_train, y_train)

    clf = Clf()
    name = Clf.__name__

    grid_search = GridSearchCV(clf, param_grid=param_grid,  cv=cv_indices)
    start = time()
    
    grid_search.fit(X_train, y_train)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time() - start, len(grid_search.cv_results_['params'])))

    report(grid_search.cv_results_, name)

    # not neccessary to do, but good to compare that result is similar than with gridsearch
    testModel = Clf(**grid_search.best_params_)
    testModel.fit(X_t, y_t)
    score = accuracy_score(y_cv, testModel.predict(X_cv))
    print()
    print("--------------------")
    print(name)
    print("Best estimator crossValidation score:", score)
    print("Parameters: {0}".format(grid_search.best_params_))
    print("--------------------")
    print()

    # Predict against test set
    y_pred_test = grid_search.predict(X_test)

    # write submission file for prediction
    y_pred_labels = list(y_label_encoder.inverse_transform(y_pred_test))
    write_submission(name, y_pred_labels)
    
    return clf, score
