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
from bertopic import BERTopic  # type: ignore
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.load_data import load_clean_data  # type: ignore


# Data Overview

st.title("🗂️ Data Overview & Cleaning")

@st.cache_data
def load_raw_data():
    return pd.read_csv("data/raw/supply_chain_project_trustpilot_advanced_merge.csv")

raw_df = load_raw_data()

st.subheader("🔎 Raw Data Preview")
st.dataframe(raw_df.head(10))

st.subheader("📐 Raw Data Shape")
st.write(f"**Rows:** {raw_df.shape[0]} | **Columns:** {raw_df.shape[1]}")


if st.checkbox("🧼 Show Missing Values in Raw Data"):
    missing = raw_df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("✅ No missing values in raw data!")
    else:
        st.dataframe(missing)

if st.checkbox("🧾 Show Raw Column Names"):
    st.write(raw_df.columns.tolist())

# Filter by Company in raw data
if st.checkbox("🏢 Filter raw data by Company"):
    company = st.selectbox("Select Company", sorted(raw_df["Company"].unique()))
    filtered = raw_df[raw_df["Company"] == company]
    st.write(f"{filtered.shape[0]} rows for **{company}**")
    st.dataframe(filtered.head(10))

# Data Cleaning + Analysis

st.markdown("---")
st.subheader("🧹 Data Cleaning + Analysis of Trustpilot Reviews")

@st.cache_data
def load_and_clean_data():
    df = raw_df.copy()
    # Remove empty headings
    df['Heading'].replace(r'^\s*$', pd.NA, regex=True, inplace=True)
    df.dropna(subset=['Heading'], inplace=True)

    stop_words = set(stopwords.words('german'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in stop_words]
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        return ' '.join(tokens)

    df['cleaned_heading'] = df['Heading'].apply(preprocess_text)
    df['tokens'] = df['Heading'].apply(lambda t: preprocess_text(t).split())
    return df

df = load_and_clean_data()

# Filter by Company in cleaned data

st.subheader("📋 Sample Headings")
st.dataframe(df[['Heading', 'cleaned_heading', 'tokens']].sample(10))


st.subheader("🔠 Top 20 Most Frequent Words")
all_words = ' '.join(df['cleaned_heading']).split()
freq_df = pd.DataFrame(Counter(all_words).most_common(20), columns=['Word','Count'])
fig_bar, ax_bar = plt.subplots()
sns.barplot(data=freq_df, x='Count', y='Word', ax=ax_bar)
ax_bar.set_title("Word Frequency")
st.pyplot(fig_bar)


# Topic modeling with BERTopic

st.markdown("---")
st.subheader("🧠 Topic Clustering with BERTopic")

@st.cache_resource
def load_or_train_topic_model(headings, model_path="models/bertopic_model.pkl"):
    if os.path.exists(model_path):
        tm = joblib.load(model_path)
        st.success("Loaded existing BERTopic model")
    else:
        tm = BERTopic(language="english", verbose=True)
        tm.fit(headings)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(tm, model_path)
        st.success("Trained and saved new BERTopic model")
    return tm

headings = df['cleaned_heading'].tolist()
with st.spinner("Detecting topics..."):
    topic_model = load_or_train_topic_model(headings)

if "topics" not in st.session_state:
    topics, _ = topic_model.transform(headings)
    st.session_state.topics = topics
else:
    topics = st.session_state.topics

df['topic'] = topics

st.subheader("📌 Topic Preview")
st.dataframe(df[['cleaned_heading','topic']].sample(10))

st.subheader("📊 Top Topics Overview")
st.write(topic_model.get_topic_info().head(10))



# Optional retrain button
retrain = st.button("🔁 Retrain BERTopic Model")

headings = df['cleaned_heading'].dropna().tolist()

# Load or compute BERTopic model and topics once per session
if retrain or "bertopic_model" not in st.session_state or "bertopic_topics" not in st.session_state:
    with st.spinner("Detecting topics using BERTopic..."):
        topic_model = load_or_train_topic_model(headings)
        topics, _ = topic_model.transform(headings)

        st.session_state["bertopic_model"] = topic_model
        st.session_state["bertopic_topics"] = topics
        st.success("Topic modeling completed and stored in session.")
else:
    topic_model = st.session_state["bertopic_model"]
    topics = st.session_state["bertopic_topics"]

# Add topics to DataFrame
df["topic"] = topics

# Topic visualization
st.subheader("📈 Topic Distribution (Top 10)")
fig = topic_model.visualize_barchart(top_n_topics=10)
st.plotly_chart(fig)
