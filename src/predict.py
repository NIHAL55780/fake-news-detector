#it takes the input from the user and predicts whether the news article is fake or real


import pickle
import os
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.strip()
    stop_words=set(stopwords.words('english'))
    words = text.split()
    words=[word for word in words if word not in stop_words]
    return " ".join(words)

def predict_news(input_text):
    
    model_path = os.path.join("models","fake_news_model.pkl")
    with open(model_path,"rb") as f:
        model = pickle.load(f)
    
    vectorizer_path = os.path.join("models","tfidf_vectorizer.pkl")
    with open(vectorizer_path,"rb") as f:
        vectorizer = pickle.load(f)
        
    print("\n Enter the news article text below :")
    user_input = input(">")
    
    cleaned = clean_text(user_input)
    
    tfidf_input = vectorizer.transform([cleaned])
    prediction = model.predict(tfidf_input)
    
    if prediction == 0:
        print("\n The news article is fake")
    else:
        print("\n The news article is real")
        
if __name__ == "__main__":
    predict_news("")