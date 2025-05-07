import streamlit as st
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from preprocessing import (  # type: ignore
    clean_text,
    tokenize_text,
    lemmatize_tokens,
)

# Ensure German stopwords are downloaded
nltk.download('stopwords')
german_stopwords = set(stopwords.words('german'))

# Load raw dataset from file
@st.cache_data
def load_data():
    path = "data/raw/supply_chain_project_trustpilot_advanced_merge.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.dropna(subset=['Heading'], inplace=True)
        return df
    else:
        st.error("CSV file not found!")
        return pd.DataFrame()

df = load_data()

# Title
st.title("🧪 Step-by-Step Text Preprocessing")

# Step-by-step preprocessing section
if not df.empty:
    selected_heading = st.selectbox("Select a sample heading from the dataset:", df['Heading'].unique())

    if st.button("Run Preprocessing"):
        st.subheader("1️⃣ Original Text")
        st.write(selected_heading)

        cleaned = clean_text(selected_heading)
        st.subheader("2️⃣ Cleaned (lowercased, punctuation and URL removed, etc.)")
        st.write(cleaned)

        tokens = tokenize_text(cleaned)
        st.subheader("3️⃣ Tokenized")
        st.write(tokens)

        # Use German stopwords
        no_stopwords = [word for word in tokens if word.lower() not in german_stopwords]
        st.subheader("4️⃣ Stopwords Removed (German)")
        st.write(no_stopwords)

        lemmatized = lemmatize_tokens(no_stopwords)
        st.subheader("5️⃣ Lemmatized")
        st.write(lemmatized)

        final_text = ' '.join(lemmatized)
        st.subheader("✅ Final Preprocessed Text")
        st.write(final_text)

else:
    st.warning("Failed to load data.")