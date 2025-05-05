
def convert_to_binary_class(df, source_column="sentiment_class", target_column="binary_class"):
    df[target_column] = df[source_column].apply(lambda x: 1 if x == "positive" else 0)
    return df