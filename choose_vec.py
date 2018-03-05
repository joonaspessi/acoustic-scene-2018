import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from utils import training_data, split_data, extract_feature

def choose_best_vec():
    train_sets_names = ["A", "B", "C", "D"]

    print("Choosing best vectorization method:")
    clf = LinearDiscriminantAnalysis()
    best_score = 0
    best_vec_method = ""
    

    for vec_method in train_sets_names:
        X_train, y_train, X_test, _ = training_data()
        X_train, X_test = extract_feature(X_train, X_test, vec_method)
        # Split Training set to predefined train and cross validation
        X_t, X_cv, y_t, y_cv, _ = split_data(X_train, y_train)
        
        model = clf.fit(X_t, y_t)
        y_pred = model.predict(X_cv)
        score = accuracy_score(y_cv, y_pred)
        
        print("Method:", vec_method, "cv accuracy:", score)
        
        if score > best_score:
            best_score = score
            best_vec_method = vec_method
        
    print("Best vectorization method:", best_vec_method, "with score of:", best_score)
    return best_vec_method

if __name__ == "__main__":
    choose_best_vec()
    exit()
