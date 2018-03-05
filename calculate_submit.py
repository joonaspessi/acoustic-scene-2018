from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer

import matplotlib

import matplotlib.pyplot as plt
import utils
import numpy as np

X, y, X_test, y_label_encoder = utils.training_data()
# X, X_test = utils.extract_feature(X, X_test, "bypass", False)

X_train, X_cv, y_train, y_cv, cv_indices = utils.split_data(X, y)
y_train = to_categorical(y_train)
y_cv = to_categorical(y_cv)
y_full = to_categorical(y_cv)

lb = LabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)
y_test = lb.transform(y_cv)
num_classes = y_train.shape[1]

X_train = np.transpose(X_train, (0, 2, 1))
X_cv = np.transpose(X_cv, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

X_full = np.transpose(X, (0, 2, 1))

X_train = X_train[..., np.newaxis]
X_cv = X_cv[..., np.newaxis]
X_test = X_test[..., np.newaxis]
X_full = X_full[..., np.newaxis]


model = load_model("trained_model.h5")
y_pred_test = model.predict_classes(X_test)

y_pred_labels = list(y_label_encoder.inverse_transform(y_pred_test))
utils.write_submission("neural_prediction", y_pred_labels)
