# Extractive Text Summarization

Project Uses Semi-supervised Learning, For detailed description please look into the project report...


Steps to run the Project:

1) run helper/preprocessing.py
    - Creates a data file(data.csv) of all paragraphs from data directory


2) Manually generate train.csv by taking some sentences(not more,  20 to 30 is enough) from the above generated
   data.csv and add labels {0, 1} by your choice(0 - NOT to be included in the summary, 1 - To be included in summary)


3) run wordscoring/main.py
    - Creates predict.csv file with all the sentences in each paragraph.
    - Generates weights from Linear Regression.
    - Predicts labels with the help of those weights and features.
    - Generates labels from textrank also.
    - Writes back labels to predict.csv file


4) Validate the results manually.

    - If the results are satisfactory(happens only 1% of times) stop the process, when ever new data comes run only
      - helper/preprocessing.py
      - wordscoring/main.py

    - Else(when results are not satisfactory)
      - We have to constantly monitor and keep on adding training data to train.csv with updated labels and train the
           model again and again.

      - For this use wordscoring/regression.py

      - i.e follow the below steps:
            - Run the algorithm(regression.py) to predict for sentences in predict.csv.
            - Validate the results manually.
            - Update the labels of the existing samples of train.csv.
            - Add some more samples to train.csv with expected labels.
            - Repeat all thses sub-steps until the results of predict.csv are satisfactory

      - After few iterations our train.csv would be very efficient so that it can be used to create a linear regression model that predicts with more accuracy.

      - Whenever new data comes, we can directly run
            - helper/preprocessing.py
            - wordscoring/main.py