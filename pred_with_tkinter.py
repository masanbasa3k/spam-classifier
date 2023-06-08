import tkinter as tk
from tkinter import messagebox
import joblib
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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

# Function to handle button click event
def classify_text():
    # Get the input text from the entry widget
    text = text_entry.get("1.0", "end-1c")
    
    # Preprocess the text
    clean_text = text_process(text)
    
    # Transform the text using CountVectorizer and TfidfTransformer
    text_dtm = vect.transform([clean_text])
    text_tfidf = tfidf_transformer.transform(text_dtm)
    
    # Perform the prediction
    prediction = loaded_model.predict(text_tfidf)
    
    # Determine the label
    label = "spam" if prediction == 1 else "ham"
    
    # Show a message box with the prediction result
    messagebox.showinfo("Prediction", f"The text is classified as: {label}")

# Create the main window
window = tk.Tk()
window.title("Spam Classifier")

# Create a text entry widget
text_entry = tk.Text(window, height=10, width=30)
text_entry.pack()

# Create a button widget
classify_button = tk.Button(window, text="Classify", command=classify_text)
classify_button.pack()

# Start the Tkinter event loop
window.mainloop()
