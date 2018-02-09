import numpy as np


def r_at_1(labels, predictions):
    return mean_recall_at_k(labels, predictions, k=1)


def r_at_2(labels, predictions):
    return mean_recall_at_k(labels, predictions, k=2)


def r_at_5(labels, predictions):
    return mean_recall_at_k(labels, predictions, k=5)


def mean_recall_at_k(labels, predictions, k):
    predictions = np.flip(np.argsort(predictions, -1), -1)[:, :k]
    flags = np.zeros_like(predictions)
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i][j] in np.arange(labels[i][j]):
                flags[i][j] = 1.
    return np.mean((np.sum(flags, -1) >= 1.).astype(float))


def rank_response(labels, predictions):
    predictions = np.flip(np.argsort(predictions, -1), -1)
    ranks = []
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i][j] in np.arange(labels[i][j]):
                ranks.append(j)
                break
    return np.mean(np.asarray(ranks).astype(float))