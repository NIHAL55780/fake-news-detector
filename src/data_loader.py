#it loads the raw data from CSV files, combines them, and saves the processed data


import pandas as pd #library that helps to read the CSV file
import os

def load_data():
    fake_path=os.path.join("data","raw","Fake.csv")  #os.path.join is used to create a path that works on any operating system for windows it is equivalent to "data\raw\Fake.csv"
    true_path=os.path.join("data","raw","True.csv")
    
    fake_df=pd.read_csv(fake_path)  #reading the CSV file and storing it in a dataframe
    true_df=pd.read_csv(true_path)
    
    fake_df["label"]=0  #adding a new column 'label' with value 0 for fake news
    true_df["label"]=1
    
    df= pd.concat([fake_df,true_df],ignore_index=True) #this line will combine the two dataframes and the ignore_index will reset the index of the new dataframe
    
    processed_path=os.path.join("data","processed","combined.csv")
    df.to_csv(processed_path,index=False)  #it will save the df to the new path without the index column to make it cleaner
    print("dataset has been loaded and combined successfully")
    print(f"saved to {processed_path}")
    
if __name__=="__main__": #should be outside the function
    load_data()