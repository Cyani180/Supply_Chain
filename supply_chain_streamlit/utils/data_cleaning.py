import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def clean_dataframe(df):
    stop_words = set(nltk.corpus.stopwords.words("german"))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        if pd.isnull(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-zA-ZäöüÄÖÜß\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
        return " ".join(tokens)

    df['cleaned_comment'] = df['Comment'].apply(preprocess) if 'Comment' in df.columns else ""
    df['cleaned_heading'] = df['Heading'].apply(preprocess) if 'Heading' in df.columns else ""

    # Text zusammenführen
    df['Text'] = df['cleaned_heading'] + " " + df['cleaned_comment']

    return df