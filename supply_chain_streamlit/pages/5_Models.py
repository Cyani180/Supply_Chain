import streamlit as st
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.load_data import load_clean_data # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🤖 Model Comparison with GridSearchCV")

# --- Load data ---
@st.cache_data
def load_data():
    return load_clean_data()

df = load_data()
X = df["cleaned_comment"]
y = df["Stars"]

# --- Vectorization ---
@st.cache_resource
def vectorize_data(X):
    vectorizer = TfidfVectorizer(max_features=3000)
    return vectorizer.fit_transform(X)

# --- Session-State for vectors ---
if "X_vec" not in st.session_state:
    st.session_state["X_vec"] = vectorize_data(X)

# --- Split into Train/Test with Session-State ---
if not all(k in st.session_state for k in ["X_train", "X_test", "y_train", "y_test"]):
    X_vec = st.session_state["X_vec"]
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
else:
    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test = st.session_state["y_test"]

# --- Model configurations with GridSearchCV parameters ---
model_configs = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"]
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    },
    "Support Vector Machine": {
        "model": SVC(),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        }
    }
}

# --- Create models directory if it does not exist ---
if not os.path.exists("models"):
    os.makedirs("models")

# --- Training and display results ---
def train_and_display_models():
    model_scores = {}
    model_results = {}

    for name, config in model_configs.items():
        st.subheader(f"🔍 GridSearchCV for {name}")
        model = config["model"]
        param_grid = config["params"]

        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        model_scores[name] = acc

        best_params = grid.best_params_
        model_results[name] = {
            "params": best_params,
            "accuracy": acc,
            "y_pred": y_pred
        }

        st.write("Best Parameters:")
        st.json(best_params)
        st.write(f"Accuracy: {acc:.4f}")

        # --- Save model ---
        # Save the best model to a file
        model_path = f"models/{name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(best_model, model_path)

        # --- Confusion Matrix ---
        st.subheader(f"📊 Confusion Matrix for {name}")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        ax_cm.set_title(f"Confusion Matrix - {name}")
        st.pyplot(fig_cm)

    st.session_state["model_scores"] = model_scores
    st.session_state["model_results"] = model_results

# --- Load results or retrain ---
if "model_scores" not in st.session_state or "model_results" not in st.session_state:
    train_and_display_models()
else:
    # Display results again
    model_scores = st.session_state["model_scores"]
    model_results = st.session_state["model_results"]

    for name, result in model_results.items():
        st.subheader(f"📊 Results for {name}")
        st.write("Best Parameters:")
        st.json(result["params"])
        st.write(f"Accuracy: {result['accuracy']:.4f}")

        # Recalculate Confusion Matrix
        cm = confusion_matrix(y_test, result["y_pred"])
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        ax_cm.set_title(f"Confusion Matrix - {name}")
        st.pyplot(fig_cm)

# --- Only if not trained yet ---
# Check if model_scores is in session_state, if not, train models
if "model_scores" not in st.session_state:
    st.session_state["model_scores"] = train_models() # type: ignore

# --- Model comparison plot ---
model_scores = st.session_state["model_scores"]
st.subheader("📊 Model Comparison by Accuracy")
fig, ax = plt.subplots()
sns.barplot(x=list(model_scores.values()), y=list(model_scores.keys()), palette="Set2", ax=ax)
ax.set_xlabel("Accuracy")
ax.set_title("Model Accuracy Comparison (GridSearchCV)")
st.pyplot(fig)









