"""
    Helper for textrank algorithm
    Author - Deepti
"""

from nltk.cluster.util import cosine_distance
import numpy as np


def get_ranks(S, lam=0.0005, damping=0.80):
    """
        computes ranks for each of the sentences

        :param S            :   similarity matrix
        :param lam          :   stops the algorithm when the difference between 2 consecutive iterations is smaller or equal to lam
        :param damping      :   with a probability of 1-damping the user will simply pick a web page at random as the next destination,
                                ignoring the link structure completely in Page Rank

        :return             :   None
    """

    #Similar to gradient descent :p
    l = len(S)
    rank_old = np.ones(l) / l
    temp = (np.ones(l) * (1 - damping)) / l

    while True:
        # Keep on updating the weights
        rank_new = temp + damping * S.T.dot(rank_old)
        delta = abs((rank_new - rank_old).sum())

        # stop when the difference between the current and previous ranks is <= lam
        if delta <= lam:
            return rank_new

        rank_old = rank_new


def cosine_similarity(sent1, sent2):
    """
        computes the cosine similarity between any two given sentences

        :param sent1    :   sentence 1 to be used to compute similarity
        :param sent2    :   sentence 2 to be used to compute similarity

        :return         :   cosine similarity between both the sentences
    """

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    # combined list of bothe the sentences
    union_list = list(set(sent1 + sent2))

    v1 = [0] * len(union_list)
    v2 = [0] * len(union_list)

    # Update the vectors : assigns 1 to the words that are present in the sentence, union and
    # 0 for the words that are not present in it
    for word in sent1:
        v1[union_list.index(word)] += 1
    for word in sent2:
        v2[union_list.index(word)] += 1

    return 1 - cosine_distance(v1, v2)


def get_similarity_matrix(sentences):
    """
        computes the similarity matrix of all the sentences of a single document

        :param sentences    :   sentences of the current document

        :return             :   similarity matrix
    """

    # Create an empty similarity matrix
    S = np.zeros((len(sentences), len(sentences)))

    for row in range(len(sentences)):
        for col in range(len(sentences)):
            if row == col:
                continue

            S[row][col] = cosine_similarity(sentences[row], sentences[col])
            #print(S[row][col])

    # normalize the matrix row-wise
    for row in range(len(S)):
        S[row] = S[row] / S[row].sum()

    return S