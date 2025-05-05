import streamlit as st
import pandas as pd
from utils.load_data import load_clean_data  # type: ignore

# Load Data
@st.cache_data
def get_data():
    return load_clean_data()

df = get_data()

# Title
st.title("🗂️ Data Overview")

# Section: Data Preview
st.subheader("🔎 Data Preview")
st.dataframe(df.head(10))

# Section: Data Shape
st.subheader("📐 Data Shape")
st.write(f"**Number of Rows:** {df.shape[0]}")
st.write(f"**Number of Columns:** {df.shape[1]}")

# Section: Missing Values
if st.checkbox("🧼 Show Missing Values"):
    st.subheader("🚨 Missing Values per Column")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("✅ No missing values!")
    else:
        st.dataframe(missing)

# Section: Column Names
if st.checkbox("🧾 Show Column Names"):
    st.subheader("📌 Column Names")
    st.write(df.columns.tolist())

# Section: Data Types
if st.checkbox("🔍 Show Data Types"):
    st.subheader("🔠 Data Types")
    st.write(df.dtypes)

# Section: Statistics
if st.checkbox("📊 Show Statistical Summary"):
    st.subheader("📈 Statistics")
    st.dataframe(df.describe(include='all'))

# Section: Stars Distribution
if st.checkbox("⭐ Show Stars Distribution"):
    st.subheader("⭐ Stars Distribution")
    st.bar_chart(df["Stars"].value_counts().sort_index())

# Section: Filter by Company
if st.checkbox("🏢 Filter by Company"):
    st.subheader("🏢 Company Filter")
    selected_company = st.selectbox("Choose a Company", sorted(df["Company"].unique()))
    filtered = df[df["Company"] == selected_company]
    st.write(f"{len(filtered)} reviews for **{selected_company}**")
    st.dataframe(filtered.head(10))

# Section: Download
st.subheader("⬇️ Download Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📄 Download CSV",
    data=csv,
    file_name="cleaned_data.csv",
    mime="text/csv"
)