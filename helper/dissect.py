from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import heapq
import itertools
import csv

threshold = 3

def get_sentences(anytext):
    sent_text = sent_tokenize(anytext)

    return sent_text


def remove_stopwords(sentence):
    words_list = word_tokenize(sentence)
    stop_words = stopwords.words('english')

    extras = [',', '.', "'s"]
    stop_words = set(stop_words + extras)

    filtered_list = [w for w in words_list if not w in stop_words]

    return filtered_list


def get_summarized_sentences(sent_scores, sentences):
    labels = [0] * len(sentences)
    summary_list = []
    id_list = []
    max_elements_threshold= heapq.nlargest(threshold, sent_scores)

    for each in max_elements_threshold:
        idx = sent_scores.index(each)
        id_list.append(idx)

    id_list = sorted(id_list)

    for idx in id_list:
        labels[idx] = 1
        summary_list.append(sentences[idx])

    return  labels, summary_list


def get_sentences_for_regression(textArray):
    total_sentences_list = [["Text"]]
    trace = 1;
    sent_idx  =[1]

    for text in textArray:
        text = text.encode('ascii', 'ignore').decode('ascii')
        sentences = get_sentences(text)
        total_sentences_list.append(sentences)
        trace = trace + len(sentences)
        sent_idx.append(trace)


    single_total_sentences_list = list(itertools.chain.from_iterable(total_sentences_list))

    export(single_total_sentences_list)

    return sent_idx, total_sentences_list


def export(sent_list):
    with open("/Users/akhila/Downloads/predict.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for sent in sent_list:
            writer.writerow([sent])


def get_encoding_labels(word_scoring_labels, sent_idx, total_sent_list):
    encoded_labels = []
    summarized_list = []
    para = 1

    for i, idx in enumerate(sent_idx):
        if(i < len(sent_idx)-1): #S hould not go for the first(heading) and last element
            temp = word_scoring_labels[idx : sent_idx[i+1]]
            #print(temp, "---------\n--------")
            max_elements_threshold = heapq.nlargest(threshold, temp)

            labels = [0] * len(temp)
            summary = []
            id_list = []

            for each in max_elements_threshold:
                id = temp.index(each)
                id_list.append(id)

            id_list = sorted(id_list)

            for id in id_list:
                labels[id] = 1
                summary.append(total_sent_list[para][id])

            para = para + 1
            encoded_labels.append(labels)
            summarized_list.append(summary)

    single_encoded_labels = list(itertools.chain.from_iterable(encoded_labels))

    return single_encoded_labels, summarized_list


def perform_comparision(summ_list1, summ_list2):
    print("Not yet Implemented")