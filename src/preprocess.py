#it preprocesses the raw data by cleaning the text and removing stopwords





import pandas as pd
import re   #importing regular expression library for text cleaning
import nltk #importing natural language toolkit for text processing
from nltk.corpus import stopwords #words like the, is, in, at, which, and on
import os 

nltk.download('stopwords')  #downloading the stopwords from nltk

def clean_text(text): #cleaning of the text take place here
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)   #removing special characters and numbers
    text = text.strip() #remove the spaces from front and back
    stop_words=set(stopwords.words('english')) #load the stopwords
    words= text.split() #split the text into words
    words=[word for word in words if word not in stop_words]   #removing the stopwords from the text
    return " ".join(words)

def preprocess_data():
    path=os.path.join("data","processed","combined.csv")
    df=pd.read_csv(path)
    df['text']=df['text'].astype(str).apply(clean_text) #apply the clean text to texts in the combined dataset
    out_path=os.path.join("data","processed","cleaned.csv")
    df.to_csv(out_path,index = False)
    print("data cleaned successfully")
    print(f"cleaned data saved to {out_path}")

if __name__=="__main__":
    preprocess_data()