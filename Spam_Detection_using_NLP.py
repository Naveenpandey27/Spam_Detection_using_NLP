# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset from a URL and store it in 'data'
data = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/spam.tsv', sep='\t')

# Display the first few rows of the dataset
data.head()

# Display the count of each label ('ham' and 'spam')
data['label'].value_counts()

# Check for missing values in the dataset and display the sum of null values for each column
data.isnull().sum()

# Display the shape (number of rows and columns) of the dataset
data.shape

# Balancing the dataset by selecting a random sample of 'ham' labels to match the number of 'spam' labels
ham = data[data['label'] == 'ham']
ham.shape

spam = data[data['label'] == 'spam']
spam.shape

ham = ham.sample(spam.shape[0])
ham.shape

# Concatenate 'ham' and 'spam' data to create a balanced dataset
data = pd.concat([ham, spam], axis=0, ignore_index=True)
data.shape

# Display a random sample of 10 rows from the balanced dataset
data.sample(10)

# Prepare for machine learning training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], 
                                                    test_size=0.2, random_state=0, 
                                                    shuffle=True, stratify=data['label'])

# Create a pipeline for text classification using TF-IDF vectorization and Random Forest Classifier
clf = Pipeline([('tfidf', TfidfVectorizer()),
               ('rfc', RandomForestClassifier(n_estimators=100, n_jobs=-1))])

# Fit the model on the training data
clf.fit(X_train, y_train)

# Predict labels on the test data
y_pred = clf.predict(X_test)

# Print a classification report with evaluation metrics
print(classification_report(y_test, y_pred))

# Use the trained model to make predictions on new text samples
clf.predict(['you have won lottery ticket worth $2000, please click here to claim',
            'hi, how are you doing today?'])

# Save the trained model to a file using pickle
import pickle
pickle.dump(clf, open('model.pkl', 'wb'))

# Load the saved model from the file
model = pickle.load(open('model.pkl', 'rb'))

# Use the loaded model to make predictions on new text samples
model.predict(['you have won lottery ticket worth $2000, please click here to claim',
            'hi, how are you doing today?'])
