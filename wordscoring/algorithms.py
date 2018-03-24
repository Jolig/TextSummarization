from nltk import pos_tag
from nltk.corpus import wordnet as wn

from math import log
from math import sqrt

def get_word_frequency(words_list, word):
    return words_list.count(word)


def word_frequency(words_list):
    score = 0

    for word in words_list:
        score = score + get_word_frequency(words_list, word)

    return score


def tf_idf(l, words_list, sentences_list):
    score = 0
    df = 0

    for word in words_list:
        tf = get_word_frequency(words_list, word)

        for sentence in sentences_list:
            df = df + get_word_frequency(sentence, word)

        temp = log(1+tf)/log(1+df)
        score = score + (l * temp)

    return score


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


def n_gram(words_list):
    score = 0
    l = len(words_list)
    n = round(max(2, l/2))
    ngram_list = []

    for i in range(2, n+1):
        pos = 0

        for word in words_list[:l - i + 1]:
            ngram_list.append(words_list[pos:pos + i])
            #print(words_list[pos:pos + i])
            pos = pos + 1

    #print(ngram_list)

    for ngram in ngram_list:
        if (ngram_list.count(ngram) > 1):
            score = score + ngram_list.count(ngram)

    return sqrt(score)


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

