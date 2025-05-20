import numpy as np


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if type(y_true) != np.ndarray or type(y_pred) != np.ndarray:
        raise ValueError
    if y_true.shape != y_pred.shape:
        raise ValueError
    sse_pred = sum((y_true - y_pred) ** 2)
    y_mean_true = y_true.mean()
    sse_mean = sum((y_true - y_mean_true) ** 2)
    r2 = 1 - (sse_pred / sse_mean)
    return r2
