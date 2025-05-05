import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('german'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # HTML-Tags entfernen
    text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s]', '', text)  # Sonderzeichen entfernen
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def clean_dataframe(df):
    df = df.copy()
    df.dropna(subset=['Comment'], inplace=True)
    df['cleaned_comment'] = df['Comment'].apply(clean_text)
    return df