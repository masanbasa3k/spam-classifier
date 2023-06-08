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

# Eğitilmiş modeli yükle
loaded_model = joblib.load('spam_classifier.joblib')

# CountVectorizer ve TfidfTransformer'ı yükle
vect = joblib.load('count_vectorizer.joblib')
tfidf_transformer = joblib.load('tfidf_transformer.joblib')

# Kullanıcıdan metin girişi al
text = input("Enter a text: ")

# Metni ön işleme yap
clean_text = text_process(text)

# Metni vektöre dönüştür
text_dtm = vect.transform([clean_text])

# Metni tf-idf değerlerine dönüştür
text_tfidf = tfidf_transformer.transform(text_dtm)

# Tahmini gerçekleştir
prediction = loaded_model.predict(text_tfidf)

# Tahmin sonucunu yazdır
label = "spam" if prediction == 1 else "ham"
print(f"predict: {label}")
