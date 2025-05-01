import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# ---  ---
df=pd.read_csv("supply_chain_project_trustpilot_advanced_merge.csv")

st.title("Project: Supply Chain")
st.sidebar.title("Table of contents")
pages=["Overview", "Webscrapping", "Data", "Visualization", "Preprocessing", "Models", "Prediction"]
page=st.sidebar.radio("Go to", pages)

# --- Load data ---
if page == pages[0] : 
  st.write("### Overview")
  st.dataframe(df.head(10))
  st.write(df.shape)
  st.dataframe(df.describe())

  if st.checkbox("Show NA") :
    st.dataframe(df.isna().sum())

  st.write(df.columns)
  df.columns = df.columns.str.strip()


# ---  ---
if page == pages[1] : 
  st.write("### Webscrapping")

  
# --- Load & clean up data ---
if page == pages[2] : 
  st.write("### Data")

  # Seiten-Titel
  st.title("Daten laden und bereinigen")

  # CSV-Datei laden
  @st.cache_data
  def load_data():
      df = pd.read_csv("supply_chain_project_trustpilot_advanced_merge.csv")
      df.columns = df.columns.str.strip()  # Entferne Whitespace in Spaltennamen
      return df

  df = load_data()

  # Vorschau
  st.subheader("Datenvorschau")
  st.dataframe(df.head(10))

  # Form: Dimensionen anzeigen
  st.markdown(f"**Form:** {df.shape[0]} Zeilen × {df.shape[1]} Spalten")

  # NA-Werte anzeigen
  if st.checkbox("Zeige fehlende Werte (NA)"):
      st.write(df.isna().sum())

  # Spaltennamen anzeigen
  if st.checkbox("Spaltennamen anzeigen"):
      st.write(df.columns.tolist())

  # Beschreibung der Daten
  if st.checkbox("Statistische Übersicht"):
      st.dataframe(df.describe())

  df = df.drop(['Unnamed: 0','Name','Company','Rating_number_customer','Comment','Invitation','Dates'], axis=1)
  st.dataframe(df.head(10))
# ---  ---
if page == pages[3] : 
  st.write("### Visualization")

# ---  ---
if page == pages[4] : 
  st.write("### Preprocessing")

# ---  ---
if page == pages[5] : 
  st.write("### Models")

# ---  ---
if page == pages[6] : 
  st.write("### Prediction")
  

  









