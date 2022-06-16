import numpy as np
import torch
import datasets
import ml_metrics


# retrieval

def MRR(retrieved, reference, k=10):
    """
    :param top_k: found documents
    :param reference: reference documents
    :return: mrr : Mean Reciprocal Rank score

    For each user u:
    . Generate list of recommendations
    . Find rank k_u of its first relevant recommendation (the first rec has rank 1)
    . Compute reciprocal rank 1/k_u

    Overall algorithm performance is mean reciprocal rank :
    $$MRR(0,U)=1/|U| \sum_{u\in U}1/k_u$$
    """
    top_k = retrieved[:k]
    k_u = []
    for i, j in enumerate(top_k):
        if j in reference:
            k_u.append(i + 1)
    mrr = sum(1 / k_u_i for k_u_i in k_u) / k
    return mrr


def ACCURACY(retrieved, reference, k=10, return_list=False):
    """
    :param top_k: found documents
    :param reference: reference documents
    :return:  acc : Accuracy of top_k
    """
    top_k = retrieved[:k]
    acc = sum(top_k_i in reference for top_k_i in top_k) / k
    if return_list :
        return acc, [top_k_i in reference for top_k_i in top_k]
    else:
        return acc

def RECALL(retrieved, reference, k=10):
    top_k = retrieved[:k]
    recall = sum(top_k_i in reference for top_k_i in top_k) / len(reference)
    return recall

def MAP(retrieved, reference, k=10):
    map = ml_metrics.mapk([reference], [retrieved], k)
    return map


# generation

def rouge(generated, reference, **kwargs):
    metric = datasets.load_metric('rouge')
    metric.add_batch(predictions=generated, references=reference)
    results = metric.compute()
    return results

def bleu(generated, reference, **kwargs):
    metric = datasets.load_metric('sacrebleu')
    metric.add_batch(predictions=generated, references=[[r] for r in reference])
    results = metric.compute()
    return results


def bert_score(generated, reference, lang="en", **kwargs):
    metric = datasets.load_metric("bertscore")
    metric.add_batch(predictions=generated, references=reference)
    results = metric.compute(lang=lang, verbose=True)
    return results

def meteor(generated, reference, **kwargs):
    metric = datasets.load_metric("meteor")
    metric.add_batch(predictions=generated, references=reference)
    results = metric.compute()
    return results


print(meteor([[1,2,3],[4,5,3,2],[4,3,2,1]], [[2,1,3],[1,4,3,2],[1,2]]))