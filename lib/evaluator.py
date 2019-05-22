import numpy as np


def logloss(y_true, y_proba):

    if y_true.ndim != 1:
        y_true = y_true.flatten()
    if y_proba.ndim != 1:
        y_proba = y_proba.flatten()

    return -(np.sum(y_true * np.log(y_proba + 1e-10) +
                    (1 - y_true) * np.log(1 - y_proba + 1e-10)) /
             y_true.shape[0])
