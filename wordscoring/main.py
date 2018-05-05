"""
Main Program that performs
1) Data Split to Sentences
2) Linear Regression
3) TextRank

Author - Akhila

"""


from wordscoring import regression as lreg
from graphscoring import main as tr
from helper import dissect

import pandas as pd

def main():

    # Training file, mostly created manually
    train_file = "/Users/akhila/Downloads/train.csv"

    # File that contains each sentence as a row and after running this program it updates the file with
    # class labels{0, 1} along with the sentences present before
    predict_file = "/Users/akhila/Downloads/predict.csv"

    # File that contains data from each document as a row
    data_file = "/Users/akhila/Downloads/data.csv"

    df = pd.read_csv(data_file)
    textArray = df["Paragraph"].values.tolist()



    # Adds sentences to the predict.csv file and returns sentences list in each document and it's starting indices
    sent_idx, sent_list_list = dissect.get_sentences_for_regression(textArray, predict_file)
    print(sent_idx)
    print(sent_list_list)


    # Runs Linear Regression model using train.csv
    lr_weights = lreg.perform_regression("train", train_file)
    print("Linear Regression Weights: ", lr_weights, "\n")


    # Predicts labels for the sentences in predict.csv file
    word_scoring_labels = lreg.perform_regression("predict", predict_file)


    # Encode top threshold labels of a paragraph to '1' and others to '0',
    # Also returns the list of summaries by wordscoring
    word_scoring_encoded_labels, word_scoring_summarized_list = dissect.get_encoding_labels(word_scoring_labels, sent_idx, sent_list_list)
    print(word_scoring_encoded_labels)
    print(word_scoring_summarized_list)

    print("*******************************************************")

    # Encode top threshold labels of a paragraph to '1' and others to '0' performed by textrank,
    # Also returns the list of summaries by textrank
    text_rank_labels, text_rank_summarized_list = tr.perform_textrank(textArray)
    print(text_rank_labels)
    print(text_rank_summarized_list)


    # Compute the similarity between the two scoring based algorithms
    print("Similarity Percentage: ",dissect.perform_comparision(word_scoring_encoded_labels, text_rank_labels))


if __name__ == '__main__':
    main()