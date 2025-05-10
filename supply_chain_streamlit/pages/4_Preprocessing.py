# Import required libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from langdetect import detect, detect_langs  # type: ignore # Language detection
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Custom modules for preprocessing
from utils.load_and_clean_data import load_and_clean_data  # type: ignore # Custom function to load data
from preprocessing import clean_text, tokenize_text  # type: ignore # Your preprocessing functions
from germalemma import GermaLemma  # type: ignore # German lemmatizer



# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizers
lemmatizer_en = WordNetLemmatizer()
lemmatizer_de = GermaLemma()

# Lemmatization function based on language
def lemmatize_tokens(tokens, lang='de'):
    if lang == 'en':
        return [lemmatizer_en.lemmatize(token) for token in tokens]
    elif lang == 'de':
        lemmatized = []
        for token in tokens:
            # Einfache Heuristik zur Bestimmung des POS-Tags
            if token.endswith('en'):
                pos = 'V'  # Verb
            elif token[0].isupper():
                pos = 'N'  # Nomen
            else:
                pos = 'ADJ'  # Adjektiv als Standard
            try:
                lemma = lemmatizer_de.find_lemma(token, pos)
            except ValueError:
                lemma = token  # Fallback, falls POS nicht unterstützt wird
            lemmatized.append(lemma)
        return lemmatized
    else:
        return tokens   # fallback: no lemmatization

# Load stopwords once and cache the result
@st.cache_data
def load_stopwords():
    german = set(stopwords.words('german'))
    english = set(stopwords.words('english'))
    return german, english

# Load stopwords into variables
german_stopwords, english_stopwords = load_stopwords()

# Try loading and preparing the dataset
try:
    df = load_and_clean_data()
    if 'Stars' in df.columns and 'Sentiment' not in df.columns:
        # Convert star ratings to sentiment (0 = negative, 1 = positive)
        df['Sentiment'] = df['Stars'].apply(lambda x: 0 if x <= 3 else 1)
except Exception as e:
    st.error(f"Error loading data: {e}")
    df = pd.DataFrame()  # fallback to empty dataframe

# App Title
st.title("🧪 Step-by-Step Text Preprocessing")

# If data is available, continue
if not df.empty:
    # Select a sample text from the dataset
    selected_text = st.selectbox("Choose an example text:", df['Heading'].unique())

    if st.button("🚀 Start Preprocessing"):
        st.subheader("1️⃣ Original Text")
        st.write(selected_text)

        # Detect language
        try:
            lang_probs = detect_langs(selected_text)
            detected_langs = [f"{l.lang.upper()} ({round(l.prob * 100)}%)" for l in lang_probs]
            st.info("🔍 Detected Language(s): " + ", ".join(detected_langs))
            lang = lang_probs[0].lang if lang_probs else None
        except Exception:
            st.warning("⚠️ Language could not be detected.")
            lang = None

        # Preprocessing options
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
            lemmatized = lemmatize_tokens(tokens, lang=lang)
            st.subheader("5️⃣ Lemmatized")
            st.write(lemmatized)
        else:
            lemmatized = tokens

        # Show final result
        final_text = ' '.join(lemmatized)
        st.subheader("✅ Final Preprocessing")
        st.write(final_text)

    # --- Sampling Technique Comparison ---
    st.markdown("---")
    st.subheader("⚖️ Sampling Technique Comparison")

    if 'Sentiment' in df.columns:
        # Choose sampling method
        sampling_option = st.radio(
            "Choose a sampling technique to visualize class distribution:",
            ("Original Data", "Oversampling", "Undersampling"),
            index=0
        )

        # Vectorize the text data
        vectorizer = TfidfVectorizer(max_features=2000)
        X_vec = vectorizer.fit_transform(df['Heading'])
        y = df['Sentiment']

        if sampling_option == "Oversampling":
            sampler = RandomOverSampler(random_state=42)
            _, y_resampled = sampler.fit_resample(X_vec, y)
            title = "📊 Distribution after Oversampling"
        elif sampling_option == "Undersampling":
            sampler = RandomUnderSampler(random_state=42)
            _, y_resampled = sampler.fit_resample(X_vec, y)
            title = "📊 Distribution after Undersampling"
        else:
            y_resampled = y
            title = "📊 Original Distribution"

        # Plot class distribution
        counter = Counter(y_resampled)
        fig, ax = plt.subplots()
        bars = ax.bar(["Negative (0)", "Positive (1)"], [counter[0], counter[1]], color=["#f44336", "#4caf50"])
        ax.set_ylabel("Number of Samples", fontsize=13)
        ax.set_title(title, fontsize=16, pad=25)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + 5, f"{height}", ha='center', fontsize=12)

        st.pyplot(fig)
    else:
        st.warning("⚠️ 'Sentiment' column is missing in the data. Sampling comparison not possible.")
else:
    st.warning("⚠️ No data loaded.")