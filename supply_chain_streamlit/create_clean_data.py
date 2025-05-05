import pandas as pd
from utils.preprocessing import clean_dataframe

# Rohdaten einlesen (Pfad ggf. anpassen)
df = pd.read_csv("data/raw/supply_chain_project_trustpilot_advanced_merge.csv")


# Bereinigung anwenden
df_clean = clean_dataframe(df)

# Datei speichern
df_clean.to_parquet("data/processed/cleaned_data.parquet")

print("Datei gespeichert: data/processed/cleaned_data.parquet")