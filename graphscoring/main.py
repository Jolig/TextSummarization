"""
    Mainly performs textrank
    Author - Deepti
"""
from helper import dissect
from graphscoring import text_rank

import itertools


summarized_list = []
labels_list = []


def summarize(maintext):
    """
        summarizes the given text using PageRank and appends all the summaries to summarized_list (globally declared list)

        :param maintext     :   Each document data

        :return             :   None
    """

    sentences = dissect.get_sentences(maintext)
    #print("------", sentences, '\n')

    S = text_rank.get_similarity_matrix(sentences)

    sent_scores = text_rank.get_ranks(S)
    #print("sent_scores: ", sent_scores, '\n')

    labels, summary = dissect.get_summarized_sentences(sent_scores.tolist(), sentences)
    labels_list.append(labels)
    summarized_list.append(summary)


def perform_textrank(textArray):
    """
        performs textrank on the entire data

        :param textArray                :   Data read from data.csv(all documents data)

        :return single_labels_list      :   All the labels list of {0, 1}
        :return summarized_list         :   List of extracted summaries
    """

    for s in textArray:
        summarize(s)

    single_labels_list = list(itertools.chain.from_iterable(labels_list))
    return single_labels_list, summarized_list






