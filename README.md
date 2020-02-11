# Disaster Tweet Prediction

This project was for a Kaggle competition named "Real Or Not? NLP With Disaster Tweets," which can be found at this link: https://www.kaggle.com/c/nlp-getting-started/overview.

## Competition Goal
Given a set of 10,000 hand-classified tweets, train a machine learning model to predict if a tweet was about a real disaster or emergency.

## Methodology
The first step was to perform extensive cleaning of the tweets' text in order to make it suitable for modeling.
This included:
* Removing numbers
* Removing punctuation
* Removing hyperlinks and HTML tags
* Converting abbreviations and slang to regular English
* Converting inflected words to their dictionary form (for example, "walks," "walking," and "walked" become "walk").

I performed experiments with and without stopword removal from the text. In the end, I kept stopwords included because the models I used had lower F1 scores (a measure of accuracy) without them.

I experimented with several models: Logistic Regression, Multinomial Naive Bayes, Random Forests, and Support Vector Machines. I learned very quickly that Random Forests are not a suitable model for this task, as there is too much noise for that model to be effective. As measured by F1 score, the Multinomial Naive Bayes model provided the most accurate predictions.

## Findings
On the training set of data, the Bayes model was 68.39% accurate in its predictions. When applied to the test set of data, it received a preliminary score of 80.78% on the Kaggle leaderboard. The preliminary score is calculated using approximately 30% of the test set of data, so this score may change.

## Repository Contents
The root folder of this repository contains the final model notebook (disaster_tweets_nlp.ipynb) and a PDF copy of the accompanying presentation I created for this project, "That Tweet Was A Disaster."

The presentation can also be seen on the Web at this link: https://www.canva.com/design/DADzc6ttcAE/meMii2-S2Pl0ZtbNoGCJYA/view?utm_content=DADzc6ttcAE&utm_campaign=designshare&utm_medium=link&utm_source=homepage_design_menu

There are 2 subfolders in this repository, "data" and "notebooks."

### The data subfolder contains 10 files:
* "raw" training data, testing data and sample submission file (train.csv, test.csv, sample_submission.csv)
* "cleaned" training data (cleaned_train.csv)
* canonicalized training data (stemmed_train.csv, lemmatized_train.csv)
* "cleaned" and canonicalized testing data (cleaned.csv)
* predictions for submission to the contest (lr_prediction_submission.csv, mnb_prediction_submission.csv, svc_prediction_submission.csv)

### The notebooks subfolder contains 3 files:
* a notebook for data exploration and cleaning (disaster_tweet_data_processing.ipynb)
* a notebook for modeling experimentation (disaster_tweet_modeling.ipynb)
* a Python script file containing some helper functions used in the notebooks (helpers.py)

## Future Steps
In order to improve my models, I will continue experimentation to find the optimal hyperparameters for the models I used.

Additionally, I plan to implement a neural net model to see if it's more effective in making these predictions.
