import pandas as pd
from sentiment_utils import polarity 

# Load cleaned data 
df = pd.read_csv("cleaned_reviews.csv")


#  Boolean baseline
def boolean_search(df, raw_query):
    aspect_raw, opinion_raw = raw_query.split(":")
    aspect_terms  = aspect_raw.strip().lower().split()
    opinion_terms = opinion_raw.strip().lower().split()

    mask_aspect  = df['cleaned_review'].apply(
        lambda txt: all(t in txt for t in aspect_terms))
    mask_opinion = df['cleaned_review'].apply(
        lambda txt: all(t in txt for t in opinion_terms))

    return df[mask_aspect & mask_opinion]


# Method 1: rating filter
def search_with_rating(df, raw_query):
    hits = boolean_search(df, raw_query)
    first_op = raw_query.split(":")[1].strip().split()[0]
    senti    = polarity(first_op)

    if senti == "positive":
        hits = hits[hits["customer_review_rating"] > 3]
    elif senti == "negative":
        hits = hits[hits["customer_review_rating"] <= 3]

    return hits.reset_index(drop=True)


# Quick comparison
if __name__ == "__main__":
    queries = [
        "audio quality: poor",
        "wifi signal: strong",
        "mouse button: click problem",
        "gps map: useful",
        "image quality: sharp",
    ]

    for q in queries:
        base = boolean_search(df, q)
        m1   = search_with_rating(df, q)
        print(f"{q:25}  Base:{len(base):4}  M1:{len(m1):4}")
