import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Configuration
ORIGINAL_CSV_PATH = "reviews.csv"
CLEANED_CSV_PATH = "cleaned_reviews.csv"

def create_cleaned_dataset():
    # Reads the raw reviews.csv, cleans the text, and saves it to
    # cleaned_reviews.csv.
    print(f"Attempting to load '{ORIGINAL_CSV_PATH}'...")
    try:
        df = pd.read_csv(ORIGINAL_CSV_PATH)
    except FileNotFoundError:
        print(f"--- FATAL ERROR ---")
        print(f"Original data file not found at '{ORIGINAL_CSV_PATH}'.")
        print("Please ensure 'reviews.csv' is in the main project directory.")
        return

    # Download necessary NLTK data if not present
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK 'stopwords'...")
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')

    # Define the cleaning function
    stop_words = set(stopwords.words('english'))
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words]
        return " ".join(tokens)

    # Apply cleaning
    print("Cleaning raw review text... (This may take a minute or two)")
    df['cleaned_review'] = df['review_text'].apply(clean_text)
    
    # Save the newly cleaned file
    df.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"Successfully created and saved '{CLEANED_CSV_PATH}'.")

if __name__ == "__main__":
    create_cleaned_dataset()
