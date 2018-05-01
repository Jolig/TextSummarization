from nltk import pos_tag
from nltk.corpus import wordnet as wn

from math import log
import itertools

from helper import dissect

def get_word_frequency(words_list, word):
    return words_list.count(word)


def word_frequency_and_tf_df(l, words_list, sentences_list):
    tf_df_score = 0
    word_freq_score = 0
    df = 0

    for word in words_list:
        tf = get_word_frequency(words_list, word)

        for sentence in sentences_list:
            sent_words_list = dissect.remove_stopwords(sentence)
            df = df + get_word_frequency(sent_words_list, word)

        temp1 = log(1+tf)/log(1+df)
        temp2 = (1+tf)/(1+df)

        tf_df_score = tf_df_score + (l * temp1)
        word_freq_score = word_freq_score + (l * temp2)

    return word_freq_score, tf_df_score


def upper_case(words_list):
    count = 0

    for word in words_list:
        if(word[0].isupper()):
            count = count + 1

    return (count/len(words_list))


def proper_noun(words_list):
    count = 0;

    for word, pos in pos_tag(words_list):
        if (pos == 'NNP'):
            count = count + 1

    return count


# def n_gram(words_list):
#     score = 0
#     l = len(words_list)
#     n = round(max(2, l/2))
#     ngram_list = []
#
#     for i in range(2, n+1):
#         pos = 0
#
#         for word in words_list[:l - i + 1]:# +1 since that would be exclusive
#             ngram_list.append(words_list[pos:pos + i])
#             #print(words_list[pos:pos + i])
#             pos = pos + 1
#
#     #print(ngram_list)
#
#     for ngram in ngram_list:
#         if (ngram_list.count(ngram) > 1):
#             score = score + ngram_list.count(ngram)
#
#     return sqrt(score)


def get_total_ngrams(sent_list):
    total_words_list= []

    for sent in sent_list:
        total_words_list.append(dissect.remove_stopwords(sent))

    single_total_words_list = list(itertools.chain.from_iterable(total_words_list))

    return get_n_grams(single_total_words_list)


def get_n_grams(words_list):
    l = len(words_list)
    ngram_list = []

    for i in range(2, l + 1):
        pos = 0

        for word in words_list[:l - i + 1]:  # +1 since that would be exclusive
            ngram_list.append(words_list[pos:pos + i])
            pos = pos + 1

    return ngram_list


def n_gram(words_list, total_n_grams):
    score = 0
    l = len(words_list)

    ngram_list = get_n_grams(words_list)

    for ngram in ngram_list:
        score = score + total_n_grams.count(ngram)

    return score - (l * (l - 1)) / 2


def lexical_similarity(words_list):
    total_similarity = 0

    for word1 in words_list:
        word_similarity = 0

        for word2 in words_list:
            if(word1 != word2):
                synset1 = wn.synsets(word1)
                synset2 = wn.synsets(word2)

                if(len(synset1) > 0 and len(synset2)):
                    temp = synset1[0].path_similarity(synset2[0])

                    if(temp != None):
                        word_similarity = word_similarity + temp
                        #print(temp)

        total_similarity = total_similarity + word_similarity
        #print(total_similarity)

    return total_similarity

