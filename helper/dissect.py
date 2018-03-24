from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords


def get_sentences(anytext):
    sent_text = sent_tokenize(anytext)

    return len(sent_text), sent_text



def remove_stopwords(sentence):
    words_list = word_tokenize(sentence)
    stop_words = stopwords.words('english')

    extras = [',', '.', "'s"]
    stop_words = set(stop_words + extras)

    filtered_list = [w for w in words_list if not w in stop_words]

    return filtered_list
