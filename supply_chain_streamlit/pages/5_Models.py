import streamlit as st
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.load_data import load_clean_data  # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.title("🤖 Model Comparison with GridSearchCV + F1‑Score Evaluation")

# ------------------------ Load and Prepare Data ------------------------
@st.cache_data
def load_data():
    return load_clean_data()

df = load_data()
X = df["cleaned_comment"]
y = df["Stars"]

# ------------------------ TF-IDF Vectorization ------------------------
@st.cache_resource
def vectorize_data(X):
    vectorizer = TfidfVectorizer(max_features=3000)
    return vectorizer.fit_transform(X)

if "X_vec" not in st.session_state:
    st.session_state["X_vec"] = vectorize_data(X)

X_vec = st.session_state["X_vec"]

# ------------------------ Train/Test Split ------------------------
if not all(k in st.session_state for k in ["X_train", "X_test", "y_train", "y_test"]):
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )
    st.session_state.update({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test})
else:
    X_train = st.session_state["X_train"]
    X_test  = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test  = st.session_state["y_test"]

# ------------------------ Model Configuration ------------------------
model_configs = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {"C": [0.1, 1, 10], "solver": ["liblinear", "lbfgs"]},
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(),
        "params": {"max_depth": [10, 20, None], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]},
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
    },
    "Support Vector Machine": {
        "model": SVC(probability=True),
        "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
    },
}

if not os.path.exists("models"):
    os.makedirs("models")

# ------------------------ Training and Display ------------------------
def train_and_display_models():
    model_scores = {}
    model_results = {}

    for name, cfg in model_configs.items():
        st.subheader(f"🔍 GridSearchCV for {name}")
        grid = GridSearchCV(cfg["model"], cfg["params"], cv=3, n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        model_scores[name] = acc
        model_results[name] = {"model": best_model, "y_pred": y_pred}

        st.write("**Best Parameters:**")
        st.json(grid.best_params_)
        st.write(f"**Accuracy:** {acc:.4f}")

        # Save the trained model
        joblib.dump(best_model, f"models/{name.replace(' ', '_').lower()}_model.pkl")

        # Confusion Matrix
        st.markdown(f"**Confusion Matrix – {name}**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        st.pyplot(fig_cm)

        # Classification Report as DataFrame
        report_dict = classification_report(
            y_test, y_pred, digits=2, zero_division=0, output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        st.markdown("**Classification Report:**")
        st.dataframe(report_df.style.format("{:.2f}"))

        # F1-Score per class
        classes = sorted(y_test.unique())
        f1_per = f1_score(y_test, y_pred, average=None, labels=classes)
        fig_f1, ax_f1 = plt.subplots()
        ax_f1.bar(classes, f1_per, color="orange")
        ax_f1.set_xlabel("Rating Class")
        ax_f1.set_ylabel("F1-Score")
        ax_f1.set_title(f"F1-Score per Class – {name}")
        st.pyplot(fig_f1)

    st.session_state["model_scores"] = model_scores
    st.session_state["model_results"] = model_results

# Run training if not already done
if "model_scores" not in st.session_state:
    train_and_display_models()

# ------------------------ Reuse Trained Models ------------------------
else:
    for name, res in st.session_state["model_results"].items():
        st.subheader(f"📊 Results for {name}")
        st.write(f"**Accuracy:** {st.session_state['model_scores'][name]:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, res["y_pred"])
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        st.pyplot(fig_cm)

        # Classification Report as DataFrame
        report_dict = classification_report(
            y_test, res["y_pred"], digits=2, zero_division=0, output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        st.markdown("**Classification Report:**")
        st.dataframe(report_df.style.format("{:.2f}"))

        # F1-Score per class
        classes = sorted(y_test.unique())
        f1_per = f1_score(y_test, res["y_pred"], average=None, labels=classes)
        fig_f1, ax_f1 = plt.subplots()
        ax_f1.bar(classes, f1_per, color="orange")
        ax_f1.set_xlabel("Rating Class")
        ax_f1.set_ylabel("F1-Score")
        ax_f1.set_title(f"F1-Score per Class – {name}")
        st.pyplot(fig_f1)

# ------------------ Summary Comparison ------------------
st.subheader("📊 Model Accuracy Comparison")
fig_acc, ax_acc = plt.subplots()
sns.barplot(
    x=list(st.session_state["model_scores"].values()),
    y=list(st.session_state["model_scores"].keys()),
    palette="Set2",
    ax=ax_acc,
)
ax_acc.set_xlabel("Accuracy")
ax_acc.set_title("Accuracy of Each Model")
st.pyplot(fig_acc)

# ------------------ Determine best model ------------------
best_model_name = max(
    st.session_state["model_scores"],
    key=lambda k: st.session_state["model_scores"][k]
)

# ------------------ Review Sentiment Distribution ------------------
st.subheader("🟢🔴 Review Sentiment Distribution")

# Combine stars into two groups: 1–3 = negative, 4–5 = positive
y_test_binary = y_test.apply(lambda x: "Negative (1–3)" if x <= 3 else "Positive (4–5)")

# Get predictions of the best model
y_pred_all = st.session_state["model_results"][best_model_name]["y_pred"]
y_pred_binary = pd.Series(y_pred_all).apply(lambda x: "Negative (1–3)" if x <= 3 else "Positive (4–5)")

# Plot the comparison of true vs. predicted sentiment groups
fig_group, ax_group = plt.subplots()
sns.countplot(x=y_test_binary, hue=y_pred_binary, palette="coolwarm", ax=ax_group)
ax_group.set_xlabel("True Sentiment")
ax_group.set_ylabel("Count")
ax_group.set_title("Predicted vs True Sentiment (Grouped)")
st.pyplot(fig_group)

# Plot overall star-group distribution
fig_bin, ax_bin = plt.subplots()
sns.countplot(x=y_test_binary, palette="coolwarm", ax=ax_bin)
ax_bin.set_title("Star Group Distribution")
ax_bin.set_xlabel("Sentiment Group")
ax_bin.set_ylabel("Count")
st.pyplot(fig_bin)