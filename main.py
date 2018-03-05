import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing
from scipy.spatial.distance import pdist

if __name__ == "__main__":

    # ---------------------------------------------------------------------
    #load data

    # X training set
    XTRAIN = np.load("data/X_train.npy")

    # X test set
    XTEST = np.load("data/X_test.npy")

    # Read
    y_train_csv = np.genfromtxt("data/y_train.csv", delimiter=',', skip_header=1, dtype=None, usecols=(1))

    # Create label encoder for Y scene_label column
    le_y = preprocessing.LabelEncoder()
    le_y.fit(y_train_csv)
    y_train = le_y.transform(y_train_csv)


    # ---------------------------------------------------------------------
    # Extract features

    # A:
    # 20040-dimensional vector from each sample
    X_train_a = XTRAIN.reshape(-1, XTRAIN.shape[1] * XTRAIN.shape[2])
    X_test_a =  XTEST.reshape(-1, XTRAIN.shape[1] * XTRAIN.shape[2])
    # B:
    # (4500,40) data matrix, each of the 4500 samples is a vector of 40 frequency bins averaged over time
    X_train_b = np.mean(XTRAIN,axis=2)
    X_test_b = np.mean(XTEST,axis=2)

    # C:
    # (4500,501) data matrix, each of the 4500 samples is a vector of 501 time points averaged frequencies
    X_train_c = np.mean(XTRAIN,axis=1)
    X_test_c = np.mean(XTEST,axis=1)

    # D:
    # (4500,780) data matrix, each of the 4500 samples is a vectorized distance matrix of the cosine/correlation/euclidean/seuclidean... 
    # distances between time point vectors on each frequency bin
    X_train_d = []
    X_test_d = []

    for index in range(np.shape(XTRAIN)[0]):
        observations = XTRAIN[index,:,:]
        dist_train = pdist(observations, 'seuclidean')
        X_train_d.append(dist_train)

    for index in range(np.shape(XTEST)[0]):
        observations = XTEST[index,:,:]
        dist_test = pdist(observations, 'seuclidean')
        X_test_d.append(dist_test)

    X_train_d = np.array(X_train_d)
    X_test_d = np.array(X_test_d)

    train_sets = [X_train_a, X_train_b, X_train_c, X_train_d]
    test_sets = [X_test_a, X_test_b, X_test_c, X_test_d]
    train_sets_names = ["A", "B", "C", "D"]


    # ---------------------------------------------------------------------
    # Split Training set to 80% test and 20% cross validation data
    X_train, X_test, y_train, y_test = train_test_split(X_train_d, y_train, test_size=0.2)

    # @todo Create loop structure for testing different classifiers with set of parameters.
    # @todo Write prediction scores to some file for further inspection

    # Classifier selection
    # clf = svm.SVC(verbose=True)
    lda_param_grid = { "solver": ["svd"] }
    clf = GridSearchCV(LinearDiscriminantAnalysis(), lda_param_grid)

    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    score = accuracy_score(y_test, y_pred)

    print("Cross validation accuracy:", score)

    # Predict against y train test
    y_pred_test = model.predict(X_test_d)

    # write submission file for prediction
    # y_pred_labels = list(le_y.inverse_transform(y_pred))
    # with open("submission.csv", "w") as fp:
    #     fp.write("Id,Scene_label\n")
    #     for i, label in enumerate(y_pred_test):
    #         fp.write("%d,%s\n" % (i, label))

exit()