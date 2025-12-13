import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer


def load_and_split_data(csv_path):
     df = pd.read_csv(csv_path)
     df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
     train_df,temp_df=train_test_split(df,test_size=0.30,random_state=42,shuffle=True,stratify=df["label"])
     val_df,test_df=train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True, stratify=temp_df["label"])
     return train_df,val_df,test_df



def prepare_features(train_df, val_df, test_df, num_cols):
  tfidf=TfidfVectorizer(max_features=5000,ngram_range=(1,2),analyzer="word")
  tfidf.fit(train_df["cleaned_text"])

  X_train_tfidf=tfidf.transform(train_df["cleaned_text"])
  X_val_tfidf=tfidf.transform(val_df["cleaned_text"])
  X_test_tfidf=tfidf.transform(test_df["cleaned_text"])
  X_train_num=train_df[num_cols].values
  X_val_num=val_df[num_cols].values
  X_test_num=test_df[num_cols].values
  X_train=hstack([X_train_tfidf,X_train_num]).tocsr()
  X_val=hstack([X_val_tfidf,X_val_num]).tocsr()
  X_test=hstack([X_test_tfidf,X_test_num]).tocsr()
  y_train= convert_labels_to_binary(train_df)
  y_val= convert_labels_to_binary(val_df)
  y_test= convert_labels_to_binary(test_df)
  return X_train,X_val,X_test,y_train,y_val,y_test


def convert_labels_to_binary(df):
  labels = df["label"].astype(str).str.strip().str.lower()
    return (labels == "ai").astype(int).values 
