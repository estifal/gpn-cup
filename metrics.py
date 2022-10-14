import numpy as np


# test git config

def precision_at_k(actual: np.ndarray,
                   predicted: np.ndarray,
                   k: int = 20) -> np.float64:
    predicted = predicted[:k]
    relevant = (predicted in actual)

    return predicted[relevant].sum()


def average_precision_at_k(actual: np.ndarray,
                           predicted: np.ndarray,
                           k: int = 20) -> float:
    predicted = predicted[:k]
    relevant = (predicted in actual)
    score = 0
    for i in range(1, k + 1):
        score += precision_at_k(actual, predicted[:i], i) * relevant[i]

    return score / k

# def mean_average_precision_at_k(actual, predicted, k=10):
#     return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
