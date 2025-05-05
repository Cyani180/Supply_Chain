import streamlit as st
import pandas as pd
import os
from textblob import TextBlob  # type: ignore
from utils.convert_to_binary_class import convert_to_binary_class # type: ignore

from preprocessing import ( # type: ignore
    clean_text,
    tokenize_text,
    remove_stop_words,
    lemmatize_tokens,
    preprocess_text,
    add_sentiment_class,
    balance_classes,
)

# Load raw data
# Check if the CSV file exists
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

# UI for step-by-step preprocessing
st.title("🧪 Step-by-Step Preprocessing of Headings")

if not df.empty:
    example_heading = st.selectbox("Choose a sample heading from the dataset:", df['Heading'].unique())

    if st.button("Start preprocessing"):

        st.subheader("1️⃣ Original Text")
        st.write(example_heading)

        cleaned = clean_text(example_heading)
        st.subheader("2️⃣ Cleaned (lowercase, removed URLs, punctuation, etc.)")
        st.write(cleaned)

        tokens = tokenize_text(cleaned)
        st.subheader("3️⃣ Tokenized")
        st.write(tokens)

        no_stopwords = remove_stop_words(tokens)
        st.subheader("4️⃣ Without Stopwords")
        st.write(no_stopwords)

        lemmatized = lemmatize_tokens(no_stopwords)
        st.subheader("5️⃣ Lemmatized")
        st.write(lemmatized)

        final_text = ' '.join(lemmatized)
        st.subheader("✅ Final Result")
        st.write(final_text)

else:
    st.warning("Failed to load data.")

# Sentiment calculation
df['sentiment'] = df['Heading'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Add sentiment class
df = add_sentiment_class(df)

# Show distribution before resampling
st.subheader("📊 Class Distribution Before Resampling")
st.bar_chart(df['sentiment_class'].value_counts())

# Resampling method selection
resample_method = st.radio("Choose a resampling method:", ["none", "oversample", "undersample"])

if resample_method != "none":
    df_balanced = balance_classes(df, label_column="sentiment_class", method=resample_method)
    st.success(f"{resample_method.capitalize()} applied successfully!")
    
    st.subheader("📊 Class Distribution After Resampling")
    st.bar_chart(df_balanced['sentiment_class'].value_counts())

    st.subheader("📋 Sample After Resampling")
    st.dataframe(df_balanced[['Heading', 'sentiment_class']].sample(10))
else:
    st.info("No resampling selected.")
    df_balanced = df

# Optional: Convert to binary classification
if st.checkbox("🎯 Convert target to binary (e.g. positive vs. rest)?"):
    df_binary = convert_to_binary_class(df_balanced)
    st.success("Target variable successfully converted to binary.")

    st.subheader("📊 Binary Class Distribution (1 = positive)")
    st.bar_chart(df_binary['binary_class'].value_counts())

    st.subheader("📋 Examples")
    st.dataframe(df_binary[['Heading', 'sentiment_class', 'binary_class']].sample(10))