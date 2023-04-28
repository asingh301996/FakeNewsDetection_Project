# FakeNewsDetection_Project
1. Project Description
Specifically, the project focuses on the task of classification of fake news, where given the title of a news article and the title of a coming news article, the goal is to classify the coming news article into one of the three categories: agreed, disagreed, or unrelated.
To accomplish this task, supervised machine learning techniques and natural language processing methods will be used for feature extraction from news articles. Various algorithms and classifier methods, including logistic regression, decision trees, random forests, support vector machines, clustering, and topic modeling, can be used.
NLP methods such as bag-of-words, n-grams, and word embeddings will be used to represent the text data in a machine-readable format. The performance of the classification model will be evaluated using standard evaluation metrics such as accuracy, precision, recall, and F1 score.
The ultimate goal of this project is to develop a reliable and effective model that can classify news articles as agreed, disagreed, or unrelated to a given fake news article, thus aiding in the detection of fake news and misinformation in social media.

2.DATA COLLECTION & OBJECTIVES
2.1.How we preprocessed the data initially
• Initially, Data preprocessing is an essential step in any machine learning project, as it involves cleaning and transforming raw data into a format that can be used by machine learning models.
• The first step in data preprocessing for this project involves removing stop words and punctuations from the news article headlines. Stop words are common words like "the", "a", "an", etc., that do not carry much meaning in the context of the text. Punctuations like commas, periods, and question marks also do not add much value to the text. Therefore, removing them can help reduce the noise in the data.
• To remove stop words and punctuations, a list of all stop words and punctuations in the English language is created using the stopword package from nltk.corpus. Then, this list is used to iterate through the data and remove these words from the headlines.
• After stop words and punctuations are removed, the headlines are tokenized and stemmed using the WhitespaceTokenizer and PorterStemmer, respectively. Tokenization involves breaking the text into words or phrases, while stemming involves reducing words to their root form. This can help reduce the number of unique words in the data and make it easier for machine learning models to analyze.
• Finally, lemmatization is performed using the WordNetLemmatizer from nltk.stem. Lemmatization is similar to stemming, but instead of reducing words to their root form, it converts them to their base or dictionary form. This can help capture the meaning of the words more accurately.
• The preprocessed data is then saved in new CSV files named train_preprocessed.csv and test_preprocessed.csv, after dropping the title1_id and title2_id columns. This data can be used for training and testing machine learning models for fake news detection.

2. Base Model Creation
2.2.1 Naïve Bayes Classifier & Logistic Regression (Approach I & II)
The labels for the test data were predicted using the trained model and stored in a pandas data frame along with their corresponding IDs. Pandas is a data manipulation library in Python that allows for easy handling of tabular data.
• Finally, the resulting data frame was saved in a CSV file for further analysis or visualization.
• Logistic regression is a type of supervised learning algorithm used for classification tasks. It works by fitting a linear regression model to the input data and then applying a sigmoid function to the output to convert it into a probability score.
• In this project, the preprocessed data was loaded and any rows with missing values were dropped to ensure the data is clean and consistent.
• The first few lines of code define a CountVectorizer object and use it to tokenize and count the words in the titles of the articles in the training set. The resulting word counts are then transformed using a TfidfTransformer to give more weight to rare words and less weight to common words.
• The features_final variable is created by concatenating the Tfidf-transformed word counts of the first and second titles of each article in the training set.
• The cosine function calculates the cosine similarity between two Tfidf-transformed word count matrices.
• The feature_similarity function applies the cosine function to the Tfidf-transformed word count matrices of the titles of each article in the training set, and returns a new DataFrame with the cosine similarities.
• The feature_train variable is created by calling the feature_similarity function on the training set DataFrame.
• The trained_labels variable is defined as the label column of the training set DataFrame.
• A Multinomial Naive Bayes model is instantiated and fitted using the feature_train and
trained_labels variables.
• The fitted Multinomial Naive Bayes model is used to predict the labels of the training set
using the predict method, and the accuracy of the predictions is calculated using the accuracy_score function.

2.4 Final Model Creation 2.4.1 LSTM (Approach III)
The first few lines of the code import the necessary libraries such as Pandas, Numpy, re, etc., and download the stopwords from the NLTK library.
o The next step reads the training and test datasets and creates an encoding for the labels. Then encoding converts the labels to numerical values that can be used by the machine learning model.
o Next function defined is clean_text(), which takes a text as input and removes all non-alphabetic characters, converts the text to lowercase, removes stop words, and returns the cleaned text. This function is applied to all text data in the dataset.
o Then one_hot_encoding(), which performs one-hot encoding of the labels.
o The next step applies the clean_text() function to the text columns of the training
and test datasets.
o The Tokenizer function from Keras is used to tokenize the text data, which is then
used to create a vocabulary and encode the text data.
o The text data is then padded to make sure all the text has the same length, which is
necessary for the machine learning model to process the data.
o The y_train and y_test variables are converted to categorical variables using the
to_categorical() function.
o The train_test_split() function is used to split the training data into training and
validation sets.
o The embedding_layer is created using the Embedding function from Keras, which is
used to convert the text data into a format that can be used by the machine learning
model.
o LSTM layer is added to the model. This layer is used to learn from the text data and
make predictions.
o Concatenate layer is added to the model to combine the output of the two LSTM
layers. The Dense layer is added to the model to perform classification on the output
of the concatenate layer.
o Model is compiled with the Adam optimizer and categorical cross-entropy loss
function. The accuracy metric is used to evaluate the model.
o The model is trained with the fit() function, and early stopping is used to prevent
overfitting.
o The trained model is used to predict the labels of the test data.
o The output is saved to a CSV file named "submission.csv," which contains the
predicted labels and their corresponding IDs. The output labels are also converted
back to their original string values.
o And we have tried plotting the
o Conclusion:
This code shows how Natural Language Processing techniques can be used to detect fake news using a machine learning model. The code uses various libraries and functions to preprocess the text data and create a model that can accurately predict the labels of pairs of headlines. The code provides a framework for building a more advanced model with better accuracy.
