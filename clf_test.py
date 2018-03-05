from utils import test_classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np

if __name__ == "__main__":
    # LDA
    lda_param_grid = {
        "solver": ["svd", "lsqr", "eigen"]
    }
    test_classifier(LinearDiscriminantAnalysis, lda_param_grid, "B")

    # SVC
    svc_param_grid = {
        "kernel": ["rbf"],
        "C": np.linspace(10e-1,10e1, num=10),
        "gamma": np.linspace(10e-3, 10e0, num=10),
        "decision_function_shape": ["ovr"]
    }
    test_classifier(svm.SVC, svc_param_grid, "B")

    # Logistic regression
    lr_param_grid = {
        "penalty": ["l2"],
        "C": np.linspace(10e-5,10e1, num=5)
    }
    test_classifier(LogisticRegression, lr_param_grid, "B")

    # RandomForestClassifier
    rf_param_grid = {
        "n_estimators": [10, 20, 30, 100, 200]
    }
    test_classifier(RandomForestClassifier, rf_param_grid, "B")

exit()
