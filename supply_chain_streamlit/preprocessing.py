import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample  
import pandas as pd
import streamlit as st

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

def tokenize_text(text):
    return word_tokenize(text)

def remove_stop_words(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stop_words(tokens)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)


def add_sentiment_class(df, polarity_column="sentiment"):
    def classify(p):
        if p < -0.1:
            return "negative"
        elif p > 0.1:
            return "positive"
        else:
            return "neutral"
    df["sentiment_class"] = df[polarity_column].apply(classify)
    return df

def balance_classes(df, label_column="sentiment_class", method="oversample"):
    classes = df[label_column].unique()
    class_dfs = [df[df[label_column] == cls] for cls in classes]
    max_size = max([len(cdf) for cdf in class_dfs])
    min_size = min([len(cdf) for cdf in class_dfs])

    balanced_dfs = []
    for cdf in class_dfs:
        if method == "oversample":
            balanced = resample(cdf, replace=True, n_samples=max_size, random_state=42)
        elif method == "undersample":
            balanced = resample(cdf, replace=False, n_samples=min_size, random_state=42)
        else:
            raise ValueError("Method must be 'oversample' or 'undersample'")
        balanced_dfs.append(balanced)

    return pd.concat(balanced_dfs)

