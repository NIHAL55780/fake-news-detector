#it is used to evaluate the performance of the trained machine learning model


import pandas as pd
import os
import pickle
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

def evaluate_model():
    data_path = os.path.join("data","processed","combined.csv")
    df=pd.read_csv(data_path)
    X = df['text']
    y = df['label']
    vectorizer_path = os.path.join("models","tfidf_vectorizer.pkl")
    with open(vectorizer_path,"rb") as f:
        vectorizer = pickle.load(f)
        
    X_tfidf = vectorizer.transform(X)
    
    model_path = os.path.join("models","fake_news_model.pkl")
    with open(model_path,"rb") as f:
        model = pickle.load(f)
        
    y_pred = model.predict(X_tfidf)
    
    print("\n Model Accuracy : ")
    print(accuracy_score(y,y_pred))
    
    print("\n Classification Report:")
    print(classification_report(y, y_pred))

    print("\n Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\n Misclassified Examples (First 5):")
    df["pred"] = y_pred
    mistakes = df[df["label"] != df["pred"]]
    print(mistakes[["text", "label", "pred"]].head())

    print("\n Probability Scores Example:")
    prob = model.predict_proba(X_tfidf[:5])
    print(prob)

if __name__ == "__main__":
    evaluate_model()