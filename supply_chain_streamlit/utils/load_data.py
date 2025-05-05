import pandas as pd

def load_clean_data():
      # If the data is used in Parquet format:
    df = pd.read_parquet("data/processed/cleaned_data.parquet")
    
    # Additional adjustments or transformations can be added here
    df = df.dropna()  # Example of removing missing values
    
    return df

