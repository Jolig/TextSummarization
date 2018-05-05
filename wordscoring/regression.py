"""
    performs linear regression by taking features as the scores of word-scoring algorithms and encoded labels{0, 1}
    Author - Akhila
"""

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
    """
        computes scores for each sentence in the sentence_list by extracting scores from each word-scoring algo

        :param sent_list    :   complete sentences list

        :return             :   list of scores list(normalized list for each sentence) of all sentences
    """

    l = len(sent_list)
    X = []

    # Compute total n_grams in the entire corpus(sentences list)
    total_n_grams = ws.get_total_ngrams(sent_list)

    for sent in sent_list:
        sent_scores = []

        words_list = dissect.remove_stopwords(sent)

        # Perform each of the specified word-scoring algotithm and append all the scores to get feature vector
        word_freq_score, tf_df_score = ws.word_frequency_and_tf_df(l, words_list, sent_list)

        sent_scores.append(word_freq_score)
        sent_scores.append(tf_df_score)
        sent_scores.append(ws.upper_case(words_list))
        sent_scores.append(ws.proper_noun(words_list))
        #sent_scores.append(ws.lexical_similarity(words_list))
        sent_scores.append(ws.n_gram(words_list, total_n_grams))

        #Normalize the scores
        sent_scores_sum = sum(sent_scores)
        sent_scores = [ele / sent_scores_sum for ele in sent_scores]

        #feature vector for linear regression
        X.append(sent_scores)

    #print(X, "\n")

    return X


def perform_regression(algo, file_name):
    """
        generic method that does both training and prediction

        :param algo         :   train/predict
        :param file_name    :   train data file/ predict data file

        :return             :   weights in case of training
                                labels in case of prediction
    """

    df = pd.read_csv(file_name)
    textArray = df["Text"].values.tolist()
    #print(textArray, "\n")

    X = get_scores(textArray)
    X = np.array(X)

    # train the model if method is used for training
    if(algo == "train"):
        y = df["Class"]
        y = np.array(y)

        lr.fit(X, y)

        return_list = []
        return_list.append(lr.coef_)
        return_list.append(lr.intercept_)

        # return the weights
        return return_list

    # predict the labels when the method is used for prediction
    if(algo == "predict"):
        res = lr.predict(X)
        res = np.array(res).tolist()
        temp = res

        textArray.insert(0, "Text")
        res.insert(0, "Prediction")

        # writes back the predicted labels along with the sentences to the predict.csv file
        rows = zip(textArray, res) 
        print(rows)
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        return temp


if __name__ == '__main__':

    train_file = "/Users/akhila/Downloads/train.csv"
    predict_file = "/Users/akhila/Downloads/predict.csv"

    # train the model
    lr_weights = perform_regression("train",train_file)
    print("Linear Regression Weights: ",lr_weights, "\n")

    # predict the labels with the help of trained model
    labels = perform_regression("predict",predict_file)
    print(labels)