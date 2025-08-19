import joblib
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Import helper functions
from sentiment_utils import polarity

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt')
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK 'stopwords'...")
    nltk.download('stopwords')

# 1. SETUP & GLOBAL VARIABLES
VEC_PATH = "tfidf.pkl"
CLF_PATH = "logreg.pkl"
STOP_WORDS = set(stopwords.words('english'))

print("Loading data and models...")
DF = pd.read_csv("cleaned_reviews.csv")
if 'review_text' not in DF.columns:
    raise ValueError("'cleaned_reviews.csv' must contain the original 'review_text' column.")

# Load the trained ML model and vectorizer
try:
    VEC = joblib.load(VEC_PATH)
    CLF = joblib.load(CLF_PATH)
    print("Loaded existing model.")
except FileNotFoundError:
    print("Training new sentiment model…")
    DF["label"] = (DF["customer_review_rating"] > 3).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        DF["cleaned_review"], DF["label"], test_size=0.2, random_state=42
    )
    VEC = TfidfVectorizer(max_features=25_000, ngram_range=(1, 2), min_df=5)
    X_train_vec = VEC.fit_transform(X_train)
    CLF = LogisticRegression(max_iter=1000, n_jobs=-1)
    CLF.fit(X_train_vec, y_train)
    print("Validation accuracy:", CLF.score(VEC.transform(X_test), y_test))
    joblib.dump(VEC, VEC_PATH)
    joblib.dump(CLF, CLF_PATH)

# 2. HELPER & MAIN SEARCH FUNCTIONS
def clean_sentence(sentence_text):
    """Cleans a single sentence to match the model's training data."""
    text = str(sentence_text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)

def search_with_ml(raw_query: str, df: pd.DataFrame = DF):
    """
    Performs a sentence-level sentiment analysis to find relevant reviews.
    """
    aspect_raw, opinion_raw = raw_query.split(":")
    aspect_phrase = aspect_raw.strip().lower()
    aspect_terms = aspect_phrase.split()
    
    # Step 1: Broad search for candidate reviews containing the aspect phrase.
    aspect_mask = df["cleaned_review"].str.contains(aspect_phrase, na=False)
    candidate_hits = df[aspect_mask]

    if candidate_hits.empty:
        return candidate_hits

    # Step 2: Verify sentiment at the sentence level.
    relevant_indices = []
    target_sentiment = polarity(opinion_raw.strip().split()[-1])
    target_label = 1 if target_sentiment == "positive" else 0

    for index, row in candidate_hits.iterrows():
        sentences = sent_tokenize(str(row['review_text']))
        for sentence in sentences:
            if all(term in sentence.lower() for term in aspect_terms):
                cleaned_sent = clean_sentence(sentence)
                prediction = CLF.predict(VEC.transform([cleaned_sent]))[0]
                if prediction == target_label:
                    relevant_indices.append(index)
                    break 

    return df.loc[list(set(relevant_indices))].reset_index(drop=True)

# 3. QUICK TEST
if __name__ == "__main__":
    test_queries = [
        "audio quality: poor",
        "wifi signal: strong",
        "mouse button: click problem",
        "gps map: useful",
        "image quality: sharp",
    ]

    print("\n--- Running Final Method 2 (Sentence-Level Analysis) ---")
    for q in test_queries:
        m2_results = search_with_ml(q)
        print(f"{q:<30} | Retrieved: {len(m2_results):4}")

