
import re
import joblib
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load saved objects
vectorizer = joblib.load('cv.pkl')        # TF-IDF vectorizer
scaler = joblib.load('scaler.pkl')        # Scaler trained on all numeric features
model = joblib.load('quora_model.pkl')    # Your trained ML model
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stopwords_set = set(stopwords.words("english"))
porter = PorterStemmer()

def decontracted(phrase):
    
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocess_question(q):
   
    q = BeautifulSoup(q, "html.parser").get_text()
    q = re.sub(r"http\S+", "", q)
    q = decontracted(q)
    q = re.sub(r"\W", " ", q)
    q = " ".join(word.lower() for word in q.split() if word not in stopwords_set)
    q = " ".join([porter.stem(word) for word in q.split()])
    return q

def common_word_count(q1, q2):
    q1_words = set(q1.split())
    q2_words = set(q2.split())
    return len(q1_words & q2_words)

def total_word_count(q1, q2):
    q1_words = set(q1.split())
    q2_words = set(q2.split())
    return len(q1_words) + len(q2_words)

def word_shared_count(q1, q2):
    
    return len(set(q1.split()) & set(q2.split()))

def create_features(q1_raw, q2_raw):
    q1 = preprocess_question(q1_raw)
    q2 = preprocess_question(q2_raw)

    q1_len = len(q1)
    q2_len = len(q2)
    q1_word_count = len(q1.split())
    q2_word_count = len(q2.split())
    common_word_cnt = common_word_count(q1, q2)
    total_words = total_word_count(q1, q2)
    word_shared = word_shared_count(q1, q2)

    numeric_features = [
        q1_len,
        q2_len,
        q1_word_count,
        q2_word_count,
        common_word_cnt,
        total_words,
        word_shared
    ]

    numeric_arr = np.array(numeric_features).reshape(1, -1)
    scaled_numeric = scaler.transform(numeric_arr)

    q1_vec = vectorizer.transform([q1]).toarray()
    q2_vec = vectorizer.transform([q2]).toarray()

    combined = np.hstack((scaled_numeric, q1_vec, q2_vec))
    return combined

def predict_duplicate(q1_raw, q2_raw):
    features = create_features(q1_raw, q2_raw)
    pred = model.predict(features)[0]
    return int(pred)  
