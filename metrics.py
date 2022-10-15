import numpy as np
from numpy import ndarray


def precision_at_k(actual: ndarray, predicted: ndarray, k: int = 20) -> float:
    predicted = predicted[:k]
    relevant = (predicted in actual)
    score = predicted[relevant].sum()

    return float(score)


def average_precision_at_k(actual: ndarray, predicted: ndarray, k: int = 20) -> float:
    predicted = predicted[:k]
    relevant: ndarray = np.in1d(predicted, actual)
    score = 0
    for i in range(1, k + 1):
        score += precision_at_k(actual, predicted[:i], i) * relevant[i]

    return score / k


def mean_average_precision_at_k(actual: ndarray,
                                predicted: ndarray,
                                k: int = 20) -> float:
    all_apk = [average_precision_at_k(a, p, k) for a, p in zip(actual, predicted)]
    score = float(np.mean(all_apk))
    return score
