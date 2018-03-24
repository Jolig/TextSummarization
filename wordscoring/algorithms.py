def word_frequency(words_list):
    score = 0

    for word in words_list:
        score = score + words_list.count(word)
    return score

def tf_idf(words_list):
    print()