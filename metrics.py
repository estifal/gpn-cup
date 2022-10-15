import numpy as np
from numpy import ndarray
from tqdm.notebook import tqdm


def precision_at_k(actual: ndarray, predicted: ndarray, k: int = 20) -> float:
    predicted = predicted[:k]
    relevant = np.in1d(predicted, actual)
    score = predicted[relevant].mean()

    return float(score)


def average_precision_at_k(actual: ndarray, predicted: ndarray, k: int = 20) -> float:
    predicted = predicted[:k]
    relevant: ndarray = np.in1d(predicted, actual)
    score = 0
    for i in range(1, k + 1):
        score += precision_at_k(actual, predicted[:i], i) * relevant[i - 1]

    return score / k


def mean_average_precision_at_k(actual: ndarray,
                                predicted: ndarray,
                                k: int = 20) -> float:
    all_apk = []
    for (a, p) in tqdm(zip(actual, predicted), total=len(actual)):
        all_apk.append(average_precision_at_k(a, p, k))
    score = float(np.mean(all_apk))
    return score
