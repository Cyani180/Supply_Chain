import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.load_data import load_clean_data  # type: ignore
from wordcloud import WordCloud  # type: ignore
from textblob import TextBlob  # type: ignore
import joblib
from bertopic import BERTopic  # type: ignore
import os

# Load cleaned data
@st.cache_data
def load_cleaned_data():
    return pd.read_parquet("data/processed/cleaned_data.parquet")

df = load_cleaned_data()

# Load or train BERTopic model
@st.cache_resource
def load_or_train_topic_model(headings, model_path="models/bertopic_model.pkl"):
    if os.path.exists(model_path):
        st.success("BERTopic model loaded ✅")
        return joblib.load(model_path)
    else:
        st.info("Training new BERTopic model... ⏳")
        model = BERTopic(language="english", verbose=True)
        model.fit(headings)
        joblib.dump(model, model_path)
        st.success("BERTopic model saved ✅")
        return model

# App title
st.title("📊 Visualization of Trustpilot Reviews")

# Rating distribution
st.header("⭐ Distribution of Star Ratings")
fig_stars, ax_stars = plt.subplots()
sns.countplot(data=df, x="Stars", order=sorted(df["Stars"].unique()), ax=ax_stars, palette="tab10")
ax_stars.set_title("Number of Reviews per Star Rating")
st.pyplot(fig_stars)

st.subheader("Table View")
star_counts = df["Stars"].value_counts().sort_index()
st.dataframe(star_counts.rename_axis("Stars").reset_index(name="Count"))

# Word cloud from cleaned comments
st.header("☁️ Word Cloud from Cleaned Comments")
if "cleaned_comment" in df.columns:
    text = " ".join(df["cleaned_comment"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wordcloud, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)
else:
    st.warning("Column 'cleaned_comment' not found.")


# Average ratings per company
st.header("🏢 Average Rating by Company")
avg_stars = df.groupby("Company")["Stars"].mean().sort_values()
fig_avg, ax_avg = plt.subplots(figsize=(8, 6))
sns.barplot(x=avg_stars.values, y=avg_stars.index, palette="coolwarm", ax=ax_avg)
ax_avg.set_xlabel("Average Rating")
ax_avg.set_ylabel("Company")
st.pyplot(fig_avg)

# Number of reviews per company
st.header("🏢 Number of Reviews by Company")
fig_comp, ax_comp = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, y="Company", order=df["Company"].value_counts().index, ax=ax_comp, palette="viridis")
ax_comp.set_xlabel("Number of Reviews")
ax_comp.set_ylabel("Company")
st.pyplot(fig_comp)

# Review activity over time
st.header("🕒 Review Activity Over Time")
df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce")
df_by_date = df.groupby("Dates").size()
fig_date, ax_date = plt.subplots(figsize=(10, 4))
df_by_date.plot(kind="line", ax=ax_date)
ax_date.set_xlabel("Date")
ax_date.set_ylabel("Number of Reviews")
st.pyplot(fig_date)