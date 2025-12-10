import pandas as pd
from sklearn.model_selection import train_test_split #is used to divide the dataset into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer #used to convert text data into numerical format
from sklearn.linear_model import LogisticRegression #importing the logistic regression model
from sklearn.metrics import accuracy_score,classification_report
import pickle
import os

def train_model():
    path = os.path.join("data","processed","cleaned.csv")
    df = pd.read_csv(path)
    df['text'] = df['text'].fillna("") #filling missing values in the 'text' column with empty strings to avoid errors during vectorization

    X=df['text']
    y=df['label']
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42  #random_state is used to ensure that the split is reproducible any number works but 42 is commonly used
    ) 
    vectorizer = TfidfVectorizer(max_features = 5000) #creates a TF-IDF vectorizer object that is used to convet the text to numerical format
    
    X_train_tfidf=vectorizer.fit_transform(X_train) #convert traing and testing data to numerical fomat 
    X_test_tfidf=vectorizer.transform(X_test)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tfidf,y_train) #we are training the model with the training data
    y_pred=model.predict(X_test_tfidf) #predicting the label (0 or 1) for the test data
    accuracy = accuracy_score(y_test,y_pred) #calculating the accuracy of the model
    print(f"Model Accuracy:{accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    model_path = os.path.join("models","fake_news_model.pkl") #path to save the trained model and vectorizer for future use
    
    vectorizer_path = os.path.join("models","tfidf_vectorizer.pkl")
    with open(model_path,"wb") as f:
        pickle.dump(model,f) #used to save the trained model to a file in binary format
    
    with open(vectorizer_path,"wb") as f:
        pickle.dump(vectorizer,f)
        
    print("Model and vectorizer saved successfully!")
        
if __name__ == "__main__":
    train_model()
