# pages/preprocessing_app.py oder main.py

import streamlit as st
import nltk
from langdetect import detect  # type: ignore
import pandas as pd

from utils.load_and_clean_data import load_and_clean_data  # <- wie oben angepasst
from preprocessing import clean_text, tokenize_text, lemmatize_tokens  # type: ignore

# Load stopwords only once
@st.cache_data
def load_stopwords():
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    german = set(stopwords.words('german'))
    english = set(stopwords.words('english'))
    return german, english

german_stopwords, english_stopwords = load_stopwords()

# Load raw data (just with minimal cleanup)
try:
    df = load_and_clean_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    df = pd.DataFrame()

# UI
st.title("🧪 Step-by-Step Text Preprocessing")

if not df.empty:
    selected_text = st.selectbox("Choose a sample text:", df['Text'].unique())

    if st.button("🚀 Start Preprocessing"):
        st.subheader("1️⃣ Original Text")
        st.write(selected_text)

        # Language detection (supporting multiple)
        try:
            from langdetect import detect_langs
            lang_probs = detect_langs(selected_text)
            detected_langs = [f"{l.lang.upper()} ({round(l.prob * 100)}%)" for l in lang_probs]
            st.info("🔍 Detected Language(s): " + ", ".join(detected_langs))
            lang = lang_probs[0].lang if lang_probs else None
        except Exception:
            st.warning("⚠️ Could not detect language.")
            lang = None

        # Processing steps
        show_clean = st.checkbox("🧼 Clean", value=True)
        show_tokenize = st.checkbox("🔤 Tokenize", value=True)
        show_stopwords = st.checkbox("🧹 Remove Stopwords", value=True)
        show_lemmatize = st.checkbox("🧬 Lemmatize", value=True)

        result = selected_text

        if show_clean:
            result = clean_text(result)
            st.subheader("2️⃣ Cleaned")
            st.write(result)

        if show_tokenize:
            tokens = tokenize_text(result)
            st.subheader("3️⃣ Tokenized")
            st.write(tokens)
        else:
            tokens = tokenize_text(result)

        if show_stopwords:
            stopwords_set = german_stopwords if lang == 'de' else english_stopwords if lang == 'en' else german_stopwords.union(english_stopwords)
            tokens = [word for word in tokens if word.lower() not in stopwords_set]
            st.subheader("4️⃣ Stopwords Removed")
            st.write(tokens)

        if show_lemmatize:
            lemmatized = lemmatize_tokens(tokens)
            st.subheader("5️⃣ Lemmatized")
            st.write(lemmatized)
        else:
            lemmatized = tokens

        final_text = ' '.join(lemmatized)
        st.subheader("✅ Final Preprocessed Text")
        st.write(final_text)

else:
    st.warning("⚠️ No data loaded.")