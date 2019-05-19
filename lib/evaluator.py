import numpy as np


def logloss(y_true, y_proba):

    return -(np.sum(y_true * np.log(y_proba + 1e-10) +
                    (1 - y_true) * np.log(1 - y_proba + 1e-10)) /
             y_true.shape[0])
