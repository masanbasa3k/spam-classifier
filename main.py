import install_requirements

import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import joblib

def text_process(mess):
    STOPWORDS = stopwords.words('english')
    mess = mess.lower()
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])

# Load the trained model
loaded_model = joblib.load('spam_classifier.joblib')

# Load CountVectorizer and TfidfTransformer
vect = joblib.load('count_vectorizer.joblib')
tfidf_transformer = joblib.load('tfidf_transformer.joblib')

# Get input text from the user
text = input("Enter a text: ")

# Preprocess the text
clean_text = text_process(text)

# Convert the text to a vector
text_dtm = vect.transform([clean_text])

# Convert the text to tf-idf values
text_tfidf = tfidf_transformer.transform(text_dtm)

# Perform the prediction
prediction = loaded_model.predict(text_tfidf)

# Print the prediction result
label = "spam" if prediction == 1 else "ham"
print(f"Prediction: {label}")
