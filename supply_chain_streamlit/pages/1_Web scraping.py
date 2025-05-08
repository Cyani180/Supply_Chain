import streamlit as st
import pandas as pd
from pathlib import Path

# Load raw data from source and save a simplified CSV version
df = pd.read_csv("data/raw/supply_chain_project_trustpilot_advanced_merge.csv")
df.to_csv("data/raw/trustpilot_scraped.csv", index=False)

# App title and description
st.title("🌐 Web Scraping Trustpilot Reviews")
st.write("Data source: Trustpilot (Example pages)")

# Path to the cleaned/simplified CSV file
DATA_PATH = Path("data/raw/trustpilot_scraped.csv")

# Cached function to load data only once unless file changes
@st.cache_data
def load_scraped_data():  
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    else:
        return pd.DataFrame()  # Return empty DataFrame if file is missing

df_scraped = load_scraped_data()

# Show warning if no data is loaded
if df_scraped.empty:
    st.warning("⚠️ No scraped data found. Run the scraping separately and save the file under:")
    st.code(str(DATA_PATH))
else:   
    # Removed: CSV download button

    # Display images in the interface
    st.image("images/reviews-on-trustpilot.png", use_container_width=True)
    st.image("images/web_explain.png", use_container_width=True)

    # Section: Search for reviews using common heading phrases
    st.subheader("🔍 Search for Reviews")

    # Extract first few words of each heading to use as selection options
    def extract_headings(df, col="Heading", n_words=5):
        return (
            df[col].dropna()
            .apply(lambda x: " ".join(x.split()[:n_words]))  # Take first n words of each heading
            .value_counts()
            .head(1000)
            .index.tolist()
        )

    # Generate heading options from the data
    headings = extract_headings(df_scraped)

    # User selects a heading phrase to filter reviews
    selected_heading = st.selectbox("Search for reviews by company name or review text", options=[""] + headings)

    # Filter and display reviews that start with the selected phrase
    if selected_heading:
        filtered_df = df_scraped[df_scraped["Heading"].str.startswith(selected_heading)]
        st.dataframe(filtered_df)

    # Section: Filter and sort data by company and star rating
    st.subheader("⚙️ Filter and Sort Data")

    # Selectbox for choosing a company
    company_filter = st.selectbox("Select a company", df_scraped["Company"].unique())

    # Slider for selecting minimum star rating
    star_filter = st.slider("Reviews from (Stars)", 1, 5, 1)

    # Filter data based on company and rating
    filtered_data = df_scraped[
        (df_scraped["Company"] == company_filter) & (df_scraped["Stars"] >= star_filter)
    ]
    st.dataframe(filtered_data)

    # Section: Paginated review loading
    st.subheader("📜 Load More Reviews")

    # Set number of rows per page
    page_size = 10
    total_rows = len(df_scraped)
    num_pages = (total_rows // page_size) + 1

    # Page selector for pagination
    page_number = st.number_input("Select page", min_value=1, max_value=num_pages, value=1)

    # Calculate start and end index for current page
    start_idx = (page_number - 1) * page_size
    end_idx = page_number * page_size

    # Display selected page of data
    st.dataframe(df_scraped.iloc[start_idx:end_idx])