from helper import dissect

from wordscoring import algorithms as ws

import pandas as pd
import numpy as np
import csv

from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

summarized_list = []

lr = LinearRegression()


def get_scores(sent_list):

    l = len(sent_list)
    X = []

    total_n_grams = ws.get_total_ngrams(sent_list)

    for sent in sent_list:
        sent_scores = []

        words_list = dissect.remove_stopwords(sent)

        word_freq_score, tf_df_score = ws.word_frequency_and_tf_df(l, words_list, sent_list)

        sent_scores.append(word_freq_score)
        sent_scores.append(tf_df_score)
        sent_scores.append(ws.upper_case(words_list))
        sent_scores.append(ws.proper_noun(words_list))
        sent_scores.append(ws.lexical_similarity(words_list))
        sent_scores.append(ws.n_gram(words_list, total_n_grams))

        #Normalization
        sent_scores_sum = sum(sent_scores)
        sent_scores = [ele / sent_scores_sum for ele in sent_scores]

        X.append(sent_scores)

    #print(X, "\n")

    return X


def perform_regression(algo, file_name):

    df = pd.read_csv(file_name)
    textArray = df["Text"].values.tolist()
    #print(textArray, "\n")

    X = get_scores(textArray)
    X = np.array(X)


    if(algo == "train"):
        y = df["Class"]
        y = np.array(y)

        lr.fit(X, y)

        return_list = []
        return_list.append(lr.coef_)
        return_list.append(lr.intercept_)

        return return_list

    if(algo == "predict"):
        res = lr.predict(X)
        res= np.array(res).tolist()
        temp = res

        textArray.insert(0, "Text")
        res.insert(0, "Prediction")

        rows = zip(textArray, res)

        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        return temp


if __name__ == '__main__':

    train_file = "/Users/akhila/Downloads/train.csv"
    predict_file = "/Users/akhila/Downloads/predict.csv"

    lr_weights = perform_regression("train",train_file)
    print("Linear Regression Weights: ",lr_weights, "\n")

    labels = perform_regression("predict",predict_file)
    print(labels)