import os

# Define the standard folder and file paths for the lexicon
LEXICON_FOLDER = "opinion-lexicon-English"
POS_PATH = os.path.join(LEXICON_FOLDER, "positive-words.txt")
NEG_PATH = os.path.join(LEXICON_FOLDER, "negative-words.txt")

def load_lexicon(pos_path=POS_PATH, neg_path=NEG_PATH):
    # Loads the positive and negative sentiment lexicons from the specified paths
    def _load(path):
        try:
            with open(path, encoding="latin-1") as f:
              # Return a set of words, ignoring comments and empty lines
                return {line.strip() for line in f if line and not line.startswith(";")}
        except FileNotFoundError:
            print(f"--- FATAL ERROR ---")
            print(f"Lexicon file not found at: '{path}'")
            print(f"Please ensure the '{LEXICON_FOLDER}' folder is in the main project directory,")
            print(f"alongside your 'Codes' and 'Outputs' folders.")
            exit() # Exit the script if the required files are missing
            
    return _load(pos_path), _load(neg_path)

# Load the lexicons when the module is imported
POS_LEX, NEG_LEX = load_lexicon()

def polarity(word: str) -> str:
    # Determines the sentiment polarity of a single word.
    # Returns 'positive', 'negative', or 'neutral'.
    w = word.lower()
    if w in POS_LEX:
        return "positive"
    if w in NEG_LEX:
        return "negative"
    return "neutral"
