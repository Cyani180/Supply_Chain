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
    return load_clean_data()

df = load_data()
X_corpus = df["cleaned_comment"]

# Fit TF-IDF vectorizer on training data
@st.cache_resource
def load_vectorizer(corpus):
    vectorizer = TfidfVectorizer(max_features=3000)
    vectorizer.fit(corpus)
    return vectorizer

vectorizer = load_vectorizer(X_corpus)

# Load models
model_paths = {
    "Logistic Regression": "models/logistic_regression_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "Support Vector Machine": "models/support_vector_machine_model.pkl"
}

# Load models into dictionary
@st.cache_resource
def load_models(paths):
    models = {}
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

models = load_models(model_paths)

# ✅ Funktion zur Vorhersage (wie gefordert)

def predict_rating(comment):
    processed = preprocess_text(comment)
    vec = vectorizer.transform([processed])
    predictions = {}
    for name, model in models.items():
        rating = model.predict(vec)[0]
        predictions[name] = int(rating)
    return predictions


# ✅ Funktion zur Benutzereingabe (wie gefordert)

def user_input_prediction():
    user_comment = input("Please insert a comment: ")
    predicted = predict_rating(user_comment)
    for model, rating in predicted.items():
        print(f"{model}: ⭐ Predicted Star Rating: {rating}")


# Streamlit UI bleibt wie gehabt


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

# Section: Preprocessing checkbox (below the text input)
preprocess = st.checkbox("Apply text preprocessing (recommended)", value=True)

# Prediction section
if st.button("🔎 Predict"):
    if not user_input.strip():
        st.warning("Please enter a valid review.")
    else:
        input_text = preprocess_text(user_input) if preprocess else user_input
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

        # Plot predictions as bar chart
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


# Info section
st.markdown("---")
st.info(
    "ℹ️ This app allows you to predict customer review ratings using three different machine learning models. "
    "It also includes a fun guessing game to test your intuition. The models are trained using TF-IDF vectorization "
    "and are based on cleaned review text. Great for demos and presentations!"
)
