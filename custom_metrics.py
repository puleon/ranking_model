import numpy as np


def r_at_1(labels, predictions):
    return mean_recall_at_k(labels, predictions, k=1)

def r_at_1_set(labels_set, predictions):
    return mean_recall_at_k_set(labels_set, predictions, k=1)

def r_at_2(labels, predictions):
    return mean_recall_at_k(labels, predictions, k=2)


def r_at_5(labels, predictions):
    return mean_recall_at_k(labels, predictions, k=5)


def map_at_1_full(labels, predictions):
    return mean_average_precision_at_k(labels, predictions, k=1, only_relevant=False)


def map_at_1_relevant(labels, predictions):
    return mean_average_precision_at_k(labels, predictions, k=1, only_relevant=True)


def mean_average_precision_at_k(labels, predictions, k, only_relevant=False):
    predictions = np.flip(np.argsort(predictions, -1), -1)
    average_prec_at_k = []
    for i in range(1,k+1):
        relevant_samples = (labels[:, i-1] == predictions[:, i-1]).astype(float)
        prec_at_k = (np.sum(labels[:, :i] == predictions[:, :i], -1) >= 1).astype(float) / float(i)
        if only_relevant:
            average_prec_at_k.append(relevant_samples * prec_at_k)
        else:
            average_prec_at_k.append(prec_at_k)
    average_prec_at_k = np.sum(np.asarray(average_prec_at_k), 0) / float(k)
    mean_average_prec_at_k = np.mean(average_prec_at_k)
    return mean_average_prec_at_k


def mean_recall_at_k(labels, predictions, k):
    labels = labels[:, :k]
    predictions = np.flip(np.argsort(predictions, -1), -1)[:, :k]
    return np.mean((np.sum((labels == predictions).astype(float), -1) >= 1).astype(float))


def mean_recall_at_k_set(labels_set, predictions, k):
    predictions = np.flip(np.argsort(predictions, -1), -1)[:, :k]
    flags = np.zeros_like(predictions)
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i][j] in labels_set[i]:
                flags[i][j] = 1.
    return np.mean((np.sum(flags, -1) >= 1.).astype(float))


def rank_response(labels, predictions):
    predictions = np.flip(np.argsort(predictions, -1), -1)
    response_id = 0
    ranks = []
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if predictions[i][j] == response_id:
                ranks.append(j)
                break
    return np.mean(np.asarray(ranks).astype(float))


def rank_response_set(labels_set, predictions):
    predictions = np.flip(np.argsort(predictions, -1), -1)
    ranks = []
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i][j] in labels_set[i]:
                ranks.append(j)
                break
    return np.mean(np.asarray(ranks).astype(float))


def rank_context(labels, predictions):
    predictions = np.flip(np.argsort(predictions, -1), -1)
    context_id = labels.shape[1] - 1
    ranks = []
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if predictions[i][j] == context_id:
                ranks.append(j)
                break
    return np.mean(np.asarray(ranks).astype(float))

def diff_top(labels, predictions):
    top_ids = np.flip(np.argsort(predictions, -1), -1)[:, 0]
    scores_top = predictions[:, top_ids]
    scores_context = predictions[:, -1]
    return np.mean((scores_top - scores_context).astype(float))


def diff_answer(labels, predictions):
    scores_response = predictions[:, 0]
    scores_context = predictions[:, -1]
    return np.mean((scores_response - scores_context).astype(float))
