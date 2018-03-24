from nltk.cluster.util import cosine_distance
import numpy as np


def get_ranks(S, lam=0.0005, damping=0.80):
    #Similar to gradient descent :p
    l = len(S)
    rank_old = np.ones(l) / l
    temp = (np.ones(l) * (1 - damping)) / l

    while True:
        rank_new = temp + damping * S.T.dot(rank_old)
        delta = abs((rank_new - rank_old).sum())

        if delta <= lam:
            return rank_new

        rank_old = rank_new


def cosine_similarity(sent1, sent2):

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    union_list = list(set(sent1 + sent2))

    v1 = [0] * len(union_list)
    v2 = [0] * len(union_list)
    for word in sent1:
        v1[union_list.index(word)] += 1
    for word in sent2:
        v2[union_list.index(word)] += 1

    return 1 - cosine_distance(v1, v2)


def get_similarity_matrix(sentences):
    S = np.zeros((len(sentences), len(sentences)))

    for row in range(len(sentences)):
        for col in range(len(sentences)):
            if row == col:
                continue
            S[row][col] = cosine_similarity(sentences[row], sentences[col])
            #print(S[row][col])

    for row in range(len(S)):
        S[row] = S[row] / S[row].sum()

    return S