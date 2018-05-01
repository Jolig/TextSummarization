from helper import dissect
from graphscoring import text_rank

import pandas as pd
import itertools


summarized_list = []
labels_list = []


def summarize(mainText):
    sentences = dissect.get_sentences(mainText)
    #print("------", sentences, '\n')

    S = text_rank.get_similarity_matrix(sentences)

    sent_scores = text_rank.get_ranks(S)
    #print("sent_scores: ", sent_scores, '\n')

    labels, summary = dissect.get_summarized_sentences(sent_scores.tolist(), sentences)
    labels_list.append(labels)
    summarized_list.append(summary)


def perform_textrank(textArray):

    for s in textArray:
        summarize(s)

    single_labels_list = list(itertools.chain.from_iterable(labels_list))
    return single_labels_list, summarized_list






