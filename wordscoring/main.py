from wordscoring import regression as lreg
from graphscoring import main as tr
from helper import dissect

import pandas as pd

def main():

    train_file = "/Users/akhila/Downloads/train.csv"
    predict_file = "/Users/akhila/Downloads/predict.csv"
    data_file = "/Users/akhila/Downloads/data.csv"

    df = pd.read_csv(data_file)
    textArray = df["Paragraph"].values.tolist()

    sent_idx, sent_list_list = dissect.get_sentences_for_regression(textArray)
    print(sent_idx)
    print(sent_list_list)

    lr_weights = lreg.perform_regression("train", train_file)
    print("Linear Regression Weights: ", lr_weights, "\n")

    word_scoring_labels = lreg.perform_regression("predict", predict_file)
    #print(word_scoring_labels)

    word_scoring_encoded_labels, word_scoring_summarized_list = dissect.get_encoding_labels(word_scoring_labels, sent_idx, sent_list_list)
    print(word_scoring_encoded_labels)
    print(word_scoring_summarized_list)

    print("*******************************************************")

    text_rank_labels, text_rank_summarized_list = tr.perform_textrank(textArray)
    print(text_rank_labels)
    print(text_rank_summarized_list)


if __name__ == '__main__':
    main()