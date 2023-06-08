import install_requirements

import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import joblib

# Set NLTK stopwords to English
STOPWORDS = stopwords.words('english')

# Function to process text by removing punctuation and stopwords
def text_process(mess):
    mess = mess.lower()
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])

# Read the CSV file containing spam data into a pandas DataFrame
df = pd.read_csv('spam.csv', encoding='latin-1')

# Remove any rows with missing values
df.dropna(how="any", inplace=True, axis=1)

# Rename the columns of the DataFrame
df.columns = ['label', 'message']

# Map the labels 'ham' and 'spam' to numerical values
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Calculate the length of each message
df['message_len'] = df.message.apply(len)

# Apply the text processing function to clean the messages
df['clean_msg'] = df.message.apply(text_process)

# Extract the clean messages (X) and corresponding labels (y) as arrays
X = df['clean_msg'].values
y = df['label_num'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create an instance of CountVectorizer and fit it to the training data
vect = CountVectorizer()
vect.fit(X_train)


# Transform the text data into a document-term matrix for training and testing sets
X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)

# Create an instance of TfidfTransformer and fit it to the training document-term matrix
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(X_train_dtm)

# Apply the TF-IDF transformation to the training document-term matrix
X_train_tfidf = tfidf_transformer.transform(X_train_dtm)


# Create a Multinomial Naive Bayes classifier and fit it to the training TF-IDF matrix and labels
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict the labels for the test TF-IDF matrix
X_test_tfidf = tfidf_transformer.transform(X_test_dtm)
y_pred_class = model.predict(X_test_tfidf)

# Print the accuracy score of the classifier
print("=======Accuracy Score===========")
print(metrics.accuracy_score(y_test, y_pred_class))

# Print the confusion matrix
print("=======Confusion Matrix===========")
print(metrics.confusion_matrix(y_test, y_pred_class))

# Save the fitted CountVectorizer for later use
joblib.dump(vect, 'count_vectorizer.joblib')
# Save the fitted TfidfTransformer for later use
joblib.dump(tfidf_transformer, 'tfidf_transformer.joblib')
# Save the trained model for future use
joblib.dump(model, 'spam_classifier.joblib')

print("Model saved.")
