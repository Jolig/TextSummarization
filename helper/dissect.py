'''
    Helper module that helps to perform intermediate operations
    Author - Devyani
'''

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import heapq
import itertools
import csv

# defines how many sentences should be present in the summary
threshold = 4


def get_sentences(anytext):
    """
        tokenizes any text into sentences

        :param anytext      :  any document text
        :return sent_text   :  list of sentences in anytext
    """

    sent_text = sent_tokenize(anytext)

    return sent_text


def remove_stopwords(sentence):
    """
        removes stopwords from a sentence and also tokenizes into words

        :param sentence         :   any sentence that needs to be tokenized to words without stop words

        :return filtered_list   :   list of words without stop words in sentence
    """

    words_list = word_tokenize(sentence)
    stop_words = stopwords.words('english')

    # extras that should also be removed
    extras = [',', '.', "'s"]
    stop_words = set(stop_words + extras)

    filtered_list = [w for w in words_list if not w in stop_words]

    return filtered_list


def get_summarized_sentences(sent_scores, sentences):
    """
        picks the sentences of that text, which has top threshold scores and label the sentences with 0's and 1's
        0 -> NOT present in the summary
        1 -> present in the summary

        :param sent_scores      :   scores retrieved from textrank algorithm
        :param sentences        :   sentences in a particular document

        :return labels          :   encoded labels with 0's and 1's
        :return summary_list    :   list of summary of a particular sentence
    """

    labels = [0] * len(sentences)
    summary_list = []
    id_list = []
    # returns the maximum threshold number of elements
    max_elements_threshold= heapq.nlargest(threshold, sent_scores)

    for each in max_elements_threshold:
        idx = sent_scores.index(each)
        id_list.append(idx)

    # sort the list to get the summary list in order
    id_list = sorted(id_list)

    for idx in id_list:
        labels[idx] = 1
        summary_list.append(sentences[idx])

    return  labels, summary_list


def get_sentences_for_regression(textArray, file_name):
    """
        splits the textArray into sentences and writes back each sentence as a row to the specified file

        :param textArray                :  Array of texts of all documents
        :param file_name                :  file that has to be created with rows as sentences in the textArray

        :return sent_idx                :  indexes of the starting sentence of each document(text in textArray) in total sentence list
        :return total_sentences_list    :  list of all the sentences in all the documents
    """

    total_sentences_list = [["Text"]]
    trace = 1;
    sent_idx  =[1]

    for text in textArray:
        text = text.encode('ascii', 'ignore').decode('ascii')
        sentences = get_sentences(text)
        total_sentences_list.append(sentences)
        trace = trace + len(sentences)
        sent_idx.append(trace)

    # Combines list of lists to single list
    single_total_sentences_list = list(itertools.chain.from_iterable(total_sentences_list))

    # exports the sentences list to the file specified
    export(single_total_sentences_list, file_name)

    return sent_idx, total_sentences_list


def export(sent_list, file_name):
    """
        exports the list specified to the file mentioned(one row for one list_item)

        :param sent_list    :   whose values has to be written into file
        :param file_name    :   file that has to be created

        :return             :   None
    """

    with open(file_name, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for sent in sent_list:
            writer.writerow([sent])


def get_encoding_labels(word_scoring_labels, sent_idx, total_sent_list):
    """
        encodes the labels to 0's and 1's from the labels we got from regression

        :param word_scoring_labels      :   predicted labels from Linear Regression
        :param sent_idx                 :   indexes of the starting sentence of each document(text in textArray) in total sentence list
        :param total_sent_list          :   list of all the sentences in all the documents

        :return single_encoded_labels   :   single list of encoded labels{0, 1} for al the sentences
        :return summarized_list         :   list of summaries(list of sentences) of each document
    """

    encoded_labels = []
    summarized_list = []
    para = 1

    for i, idx in enumerate(sent_idx):
        if(i < len(sent_idx)-1): #S hould not go for the first(heading) and last element
            temp = word_scoring_labels[idx : sent_idx[i+1]]
            #print(temp, "---------\n--------")

            # returns the maximum threshold number of elements
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

    # combines list of lists to a single list
    single_encoded_labels = list(itertools.chain.from_iterable(encoded_labels))

    return single_encoded_labels, summarized_list


def perform_comparision(summ_list1, summ_list2):
    """
        Compares both the lists and tells how much percent they matched

        :param summ_list1      :   list 1 to be compared(word-scoring encoded labels)
        :param summ_list2      :   list 2 to be compared(textrank encoded labels)

        :return comp_perct     :   Comparision percent
    """

    common_list = [i for i, j in zip(summ_list1, summ_list2) if i == j]
    cmp_perct = (len(common_list)/len(summ_list1))*100

    return cmp_perct