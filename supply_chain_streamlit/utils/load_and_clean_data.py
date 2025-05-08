import pandas as pd
import os
from .data_cleaning import clean_dataframe  # type: ignore # Verweis auf eine andere Datei in utils

def load_raw_data():
    path = "data/raw/supply_chain_project_trustpilot_advanced_merge.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else:
        raise FileNotFoundError(f"Die Datei {path} wurde nicht gefunden!")

def load_and_clean_data():
    raw_df = load_raw_data()
    if raw_df.empty:
        return pd.DataFrame()

    # Bereinigung der Daten
    df_cleaned = clean_dataframe(raw_df)

    # Sicherstellen, dass 'Text' vorhanden ist
    if "Text" not in df_cleaned.columns:
        raise ValueError("Spalte 'Text' fehlt nach der Bereinigung!")

    return df_cleaned