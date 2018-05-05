"""
    implementation of all the word-scoring algorithms
"""

from nltk import pos_tag
from nltk.corpus import wordnet as wn

from math import log
import itertools

from helper import dissect

def get_word_frequency(words_list, word):
    """
        computes the word frequency in a word_list

        :param words_list   :   list of words where word-frequency has to be computed
        :param word         :   word for which frequency has to be found

        :return             :   How many times that word appeared in the word_list
    """

    return words_list.count(word)


def word_frequency_and_tf_df(l, words_list, sentences_list):
    """
        computes word_frequency and tf_idf of each word w.r.t to the entire words in the corpus(sentences_list)

        :param l                    :   total number of sentences in the entire sentences_list
        :param words_list           :   list of all the words in the current sentence
        :param sentences_list       :   total sentences in the entire corpus

        :return word_freq_score     :   score obtained from word_frequency
        :return tf_df_score         :   score obtained from tf-idf
    """

    tf_df_score = 0
    word_freq_score = 0
    df = 0

    for word in words_list:
        tf = get_word_frequency(words_list, word)

        for sentence in sentences_list:
            sent_words_list = dissect.remove_stopwords(sentence)
            # document frequency
            df = df + get_word_frequency(sent_words_list, word)

        # tf_df score
        temp1 = log(1+tf)/log(1+df)
        # word_frequency score
        temp2 = (1+tf)/(1+df)

        # add the scores of each word to get sentence score
        tf_df_score = tf_df_score + (l * temp1)
        word_freq_score = word_freq_score + temp2

    return word_freq_score, tf_df_score


def upper_case(words_list):
    """
        computes the number of words started with Upper case in the current sentence

        :param words_list   :   list of words of current sentence

        :return             :   count of words started with Upper case
    """

    count = 0

    for word in words_list:
        if(word[0].isupper()):
            count = count + 1

    return (count)


def proper_noun(words_list):
    """
        computes the number of proper nouns in the current sentence

        :param words_list   :   list of words of current sentence

        :return             :   count of words which are pronouns
    """

    count = 0

    for word, pos in pos_tag(words_list):
        if (pos == 'NNP'):
            count = count + 1

    return count


def get_total_ngrams(sent_list):
    """
        finds n_grams of all the words in the corpus(total sentences)

        :param sent_list    :   total sentences list of all the documents

        :return             :   n_grams for the sent_list(by calling get_n_grams)
    """

    total_words_list= []

    for sent in sent_list:
        total_words_list.append(dissect.remove_stopwords(sent))

    single_total_words_list = list(itertools.chain.from_iterable(total_words_list))

    return get_n_grams(single_total_words_list)


def get_n_grams(words_list):
    """
        finds the n_grams with a threshold on 'l' for the specified list

        :param words_list       :   any set of words_list

        :return ngram_list      :   n_grams for the words_list
    """

    l = len(words_list)

    # Assumed that the probability of  more than half of the sentence is repeated in the document is very very less
    # so found only (n/2) grams for each sentence
    n = round(max(2, l / 2))

    ngram_list = []

    for i in range(2, n + 1):
        pos = 0

        for word in words_list[:l - i + 1]:  # +1 since that would be exclusive
            ngram_list.append(words_list[pos:pos + i])
            pos = pos + 1

    return ngram_list


def n_gram(words_list, total_n_grams):
    """
        computes n_gram score, i.e how many grams of the current sentence is matched with the grams of the total corpus

        :param words_list           :   current sentence list of words
        :param total_n_grams        :   list of total n_grams in the corpus

        :return                     :   n_gram score
    """

    score = 0
    l = len(words_list)
    n = round(max(2, l / 2))

    ngram_list = get_n_grams(words_list)

    for ngram in ngram_list:
        score = score + total_n_grams.count(ngram)

    # substract the n_grams corresponds to the current sentence added again in the total_n_grams list
    return score - (n * (n - 1)) / 2


def lexical_similarity(words_list):
    """
        computes the lexical_similarity scores of each of the sentence comparing it's words

        :param words_list   :   list of the words of the current sentence

        :return             :   lexical_similarity score
    """

    total_similarity = 0

    for word1 in words_list:
        word_similarity = 0

        for word2 in words_list:
            if(word1 != word2):
                synset1 = wn.synsets(word1)
                synset2 = wn.synsets(word2)

                if(len(synset1) > 0 and len(synset2)):
                    # Find the similarity between the words
                    temp = synset1[0].path_similarity(synset2[0])

                    if(temp != None):
                        word_similarity = word_similarity + temp
                        #print(temp)

        # add scores of each word to get the sentence score
        total_similarity = total_similarity + word_similarity
        #print(total_similarity)

    return total_similarity

