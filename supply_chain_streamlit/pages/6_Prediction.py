import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocess_text  # custom preprocessing
from utils.load_data import load_clean_data  # clean dataset loader

st.set_page_config(page_title="Review Rating Prediction", layout="wide")
st.title("🔮 Predict Review Rating")

# Load the dataset to train TF-IDF and for the game
@st.cache_data
def load_data():
    path = "data/raw/supply_chain_project_trustpilot_advanced_merge.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)

        # Wenn noch nicht vorhanden, kombiniere die 'Heading' und 'Comment' Spalten zu 'Text'
        if 'Text' not in df.columns:
            df['Text'] = df['Heading'] + ' ' + df['Comment']

        # Überprüfen, ob die 'Text' Spalte existiert und NaN-Werte entfernen
        if 'Text' in df.columns:
            df.dropna(subset=['Text'], inplace=True)  # Entfernen der Zeilen mit NaN in der 'Text' Spalte
        else:
            st.error("Spalte 'Text' konnte nicht erstellt werden.")
            return pd.DataFrame()  # Rückgabe eines leeren DataFrames, wenn 'Text' nicht existiert

        return df
    else:
        st.error("CSV-Datei nicht gefunden!")
        return pd.DataFrame()

# Load data
df = load_data()
if df.empty:
    st.error("Fehler beim Laden der Daten.")
else:
    # Sicherstellen, dass die Spalte 'Text' vorhanden ist, bevor wir weiter arbeiten
    if 'Text' in df.columns:
        X_corpus = df['Text']  # Benutze die kombinierte 'Text' Spalte für die TF-IDF-Transformation
    else:
        st.error("Fehler: 'Text' Spalte existiert nicht.")
        X_corpus = []  # Leerer Korpus, falls Text nicht existiert

# Fit TF-IDF vectorizer on training data
@st.cache_resource
def load_vectorizer(corpus):
    vectorizer = TfidfVectorizer(max_features=3000)
    vectorizer.fit(corpus)
    return vectorizer

if X_corpus:
    vectorizer = load_vectorizer(X_corpus)
else:
    vectorizer = None

# Load models
model_paths = {
    "Logistic Regression": "models/logistic_regression_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "Support Vector Machine": "models/support_vector_machine_model.pkl"
}

@st.cache_resource
def load_models(paths):
    models = {}
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

models = load_models(model_paths)

# Section: Random example sentences
st.subheader("📝 Choose a sample review (optional)")
sample_sentences = [
    "Die Lieferung war schnell und das Produkt hat meine Erwartungen übertroffen.",
    "Schrecklicher Kundenservice und ein kaputter Artikel.",
    "Es war okay, nichts Besonderes.",
    "Fantastische Erfahrung von Anfang bis Ende!",
    "Verspätete Lieferung und keine Reaktion vom Support."
]
selected_sentence = st.selectbox("Pick a sentence or write your own below:", options=[""] + sample_sentences)

# Section: Text input
user_input = st.text_area("✍️ Enter a customer review to predict the star rating:", value=selected_sentence)

# Section: Preprocessing checkbox
preprocess = st.checkbox("Apply text preprocessing (recommended)", value=True)

# Prediction section
if st.button("🔎 Predict"):
    if not user_input.strip():
        st.warning("Please enter a valid review.")
    else:
        input_text = preprocess_text(user_input) if preprocess else user_input
        if vectorizer:
            user_vector = vectorizer.transform([input_text])

            st.subheader("📈 Predictions from Different Models")

            ratings = []
            model_names = []
            colors = ['#4caf50', '#f44336', '#2196f3']  # Green, Red, Blue

            for idx, (model_name, model) in enumerate(models.items()):
                prediction = model.predict(user_vector)[0]
                ratings.append(prediction)
                model_names.append(model_name)
                st.success(f"{model_name}: ⭐ Predicted Rating: {int(prediction)}")

            # Plot predictions
            st.subheader("📊 Comparison Plot")
            fig, ax = plt.subplots()
            bars = ax.bar(model_names, ratings, color=colors)
            ax.set_ylabel("Predicted Rating", fontsize=14)
            ax.set_ylim(0, 5)
            ax.set_title("Model Predictions", fontsize=16)
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, int(yval), ha='center', fontsize=12)

            st.pyplot(fig)
        else:
            st.error("Fehler: Vektorizer konnte nicht geladen werden.")

# Info section
st.markdown("---")
st.info(
    "ℹ️ This app allows you to predict customer review ratings using three different machine learning models. "
    "It also includes a fun guessing game to test your intuition. The models are trained using TF-IDF vectorization "
    "and are based on cleaned review text. Great for demos and presentations!"
)