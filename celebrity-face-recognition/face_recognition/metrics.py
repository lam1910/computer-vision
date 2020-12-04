import numpy as np


def top_k_accuracy(y_true, y_pred, k):
    top_k = np.argpartition(y_pred, -1 * k, axis=1)[:, (-1 * k):]
    return np.sum(y_true == top_k)
