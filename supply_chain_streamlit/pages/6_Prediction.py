import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocess_text  # custom preprocessing
from utils.load_data import load_clean_data  # clean dataset loader

st.set_page_config(page_title="Review Sentiment Prediction", layout="wide")
st.title("🔮 Predict Review Sentiment")

# Load the dataset to train TF-IDF and for the game
@st.cache_data
def load_data():
    path = "data/raw/supply_chain_project_trustpilot_advanced_merge.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)

        if 'Text' not in df.columns:
            df['Text'] = df['Heading'] + ' ' + df['Comment']

        if 'Text' in df.columns:
            df.dropna(subset=['Text'], inplace=True)
            df = df[df['Text'].str.strip().astype(bool)]
        else:
            st.error("Column 'Text' could not be created.")
            return pd.DataFrame()

        if 'Stars' in df.columns:
            df['Sentiment'] = df['Stars'].apply(lambda x: 0 if x <= 3 else 1)

        return df
    else:
        st.error("CSV file not found!")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.error("Error loading data.")
else:
    X_corpus = df['Text'] if 'Text' in df.columns else pd.Series(dtype=str)

@st.cache_resource
def load_vectorizer(corpus: pd.Series):
    vectorizer = TfidfVectorizer(max_features=3000)
    vectorizer.fit(corpus)
    return vectorizer

if not X_corpus.empty:
    vectorizer = load_vectorizer(X_corpus)
else:
    vectorizer = None

model_paths = {
    "Logistic Regression": "models/logistic_regression_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "Support Vector Machine": "models/support_vector_machine_model.pkl"
}

@st.cache_resource
def load_models(paths: dict):
    models = {}
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

models = load_models(model_paths)

# Sample reviews
st.subheader("📝 Choose a sample review (optional)")
sample_sentences = [
    "Die Lieferung war schnell und das Produkt hat meine Erwartungen übertroffen.",
    "Schrecklicher Kundenservice und ein kaputter Artikel.",
    "Es war okay, nichts Besonderes.",
    "Fantastische Erfahrung von Anfang bis Ende!",
    "Verspätete Lieferung und keine Reaktion vom Support.",
    "Nie wieder, bin sehr enttäuscht.",
    "Jederzeit wieder hat mir sehr gut gefallen.",
    "Nur Probleme gehabt, nicht wieder!",
    "Hab was für meine Tochter bestellt und hat geklappt."
]
selected_sentence = st.selectbox("Pick a sentence or write your own below:", options=[""] + sample_sentences)

# User input
user_input = st.text_area("✍️ Enter a customer review to predict sentiment:", value=selected_sentence)

# Preprocessing option
preprocess = st.checkbox("Apply text preprocessing (recommended)", value=True)

# Prediction
if st.button("🔎 Predict"):
    if not user_input.strip():
        st.warning("Please enter a valid review.")
    else:
        input_text = preprocess_text(user_input) if preprocess else user_input
        if vectorizer:
            user_vector = vectorizer.transform([input_text])

            st.subheader("📈 Predictions from Different Models")

            sentiments = []
            model_names = []
            values = []
            colors = ['#f44336', '#4caf50']  # Red=neg, Green=pos

            for model_name, model in models.items():
                raw_pred = model.predict(user_vector)[0]
                pred_bin = 0 if raw_pred <= 3 else 1
                label = "Negative (1–3 Sterne)" if pred_bin == 0 else "Positive (4–5 Sterne)"
                sentiments.append(label)
                model_names.append(model_name)
                values.append(pred_bin)
                st.success(f"{model_name}: 💬 Predicted Sentiment: **{label}**")

            # Improved Plot
            st.subheader("📊 Sentiment Prediction Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(model_names, values, color=[colors[v] for v in values])

            ax.set_title("Model Sentiment Predictions", fontsize=18, pad=20)
            ax.set_ylabel("Sentiment (0 = Negativ, 1 = Positiv)", fontsize=14)
            ax.set_ylim(-0.1, 1.2)
            ax.set_yticks([0, 1])
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.08, str(val), ha='center', va='bottom', fontsize=12)

            st.pyplot(fig)
        else:
            st.error("Error: Vectorizer could not be loaded.")

st.markdown("---")
st.info(
    "ℹ️ This app allows you to predict customer review sentiment (Negative vs Positive) using three different machine learning models."
)