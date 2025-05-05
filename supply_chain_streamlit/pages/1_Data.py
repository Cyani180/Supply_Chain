import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from textblob import TextBlob
from bertopic import BERTopic # type: ignore
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Helper Function: Load or Train BERTopic 
@st.cache_resource
def load_or_train_topic_model(headings, model_path="models/bertopic_model.pkl"):
    if os.path.exists(model_path):
        topic_model = joblib.load(model_path)
        st.success("BERTopic model loaded ✅")
    else:
        st.info("Training new BERTopic model... ⏳")
        topic_model = BERTopic(language="english", verbose=True)
        topic_model.fit(headings)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(topic_model, model_path)
        st.success("Model saved to 'models/bertopic_model.pkl' ✅")
    return topic_model

st.title("🧹 Data Cleaning + Analysis of Trustpilot Reviews")

# Load & Clean Data
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("data/raw/supply_chain_project_trustpilot_advanced_merge.csv")
    df_original = df.copy()

    # Remove empty headings
    df['Heading'].replace(r'^\s*$', pd.NA, regex=True, inplace=True)
    df.dropna(subset=['Heading'], inplace=True)

    # Text cleaning & preprocessing
    stop_words = set(stopwords.words('german'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)     # Remove special characters
        tokens = word_tokenize(text)                # Tokenization
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        tokens = [lemmatizer.lemmatize(word) for word in tokens]      # Lemmatization
        return ' '.join(tokens)

    def get_tokens(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    df['cleaned_heading'] = df['Heading'].apply(preprocess_text)
    df['tokens'] = df['Heading'].apply(get_tokens)
    return df

df = load_and_clean_data()

# Previews + Sentiment Analysis
st.subheader("📋 Sample Headings")
st.dataframe(df[['Heading', 'cleaned_heading', 'tokens']].sample(10))

st.subheader("💬 Sentiment Analysis")
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment'] = df['cleaned_heading'].apply(get_sentiment)

fig_sent, ax_sent = plt.subplots()
sns.histplot(df['sentiment'], bins=30, kde=True, ax=ax_sent)
ax_sent.set_title("Sentiment Distribution (Polarity from -1 = negative to +1 = positive)")
st.pyplot(fig_sent)

# Most Frequent Words
st.subheader("🔠 Most Frequent Words in Cleaned Headings")
all_words = ' '.join(df['cleaned_heading'].dropna()).split()
word_counts = Counter(all_words)
most_common_df = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Count'])

fig_bar, ax_bar = plt.subplots()
sns.barplot(data=most_common_df, x='Count', y='Word', ax=ax_bar)
ax_bar.set_title("Top 20 Most Frequent Words")
st.pyplot(fig_bar)

# Topic Clustering with BERTopic
st.subheader("🧠 Topic Clustering with BERTopic")

headings = df['cleaned_heading'].dropna().tolist()

with st.spinner("Detecting topics using BERTopic..."):
    topic_model = load_or_train_topic_model(headings)

# Compute topics only once
if "topics" not in st.session_state:
    with st.spinner("Computing topic assignments..."):
        topics, _ = topic_model.transform(headings)
        st.session_state.topics = topics
        st.session_state.topic_model = topic_model
        df['topic'] = topics
else:
    topics = st.session_state.topics
    topic_model = st.session_state.topic_model
    df['topic'] = topics

# Display example topics
st.subheader("📌 Topic Preview")
st.dataframe(df[['cleaned_heading', 'topic']].sample(10))

# Overview of most frequent topics
st.subheader("📊 Top Topics Overview")
st.write(topic_model.get_topic_info().head(10))