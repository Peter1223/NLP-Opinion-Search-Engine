import pandas as pd
import os

# 'cleaned_reviews.csv' is in the same directory
try:
    df = pd.read_csv("cleaned_reviews.csv")
except FileNotFoundError:
    print("Error: 'cleaned_reviews.csv' not found.")
    exit()

# HELPER FUNCTION TO PARSE QUERY
def parse_query(raw_query):
    """Splits a raw query into aspect and opinion terms."""
    try:
        aspect_raw, opinion_raw = raw_query.split(":")
    except ValueError:
        raise ValueError("Query must contain exactly one ':'")
    
    aspect_terms = aspect_raw.strip().lower().split()
    opinion_terms = opinion_raw.strip().lower().split()
    return aspect_terms, opinion_terms

# 3 BASELINE TESTS

def run_baseline_test1(df, raw_query):
    # Test 1: Boolean Aspect Term Retrieval  
    # Retrieves reviews that contain AT LEAST ONE aspect word. 
    aspect_terms, _ = parse_query(raw_query)
    
    mask = df['cleaned_review'].apply(
        lambda txt: any(term in str(txt) for term in aspect_terms)
    )
    return df[mask]

def run_baseline_test2(df, raw_query):
    # Test 2: Aspect AND Opinion Match
    # Retrieves reviews with AT LEAST ONE aspect word AND AT LEAST ONE opinion word.
    
    aspect_terms, opinion_terms = parse_query(raw_query)
    
    mask_aspect = df['cleaned_review'].apply(
        lambda txt: any(term in str(txt) for term in aspect_terms)
    )
    mask_opinion = df['cleaned_review'].apply(
        lambda txt: any(term in str(txt) for term in opinion_terms)
    )
    return df[mask_aspect & mask_opinion]

def run_baseline_test3(df, raw_query):
    # Test 3: Aspect OR Opinion Match
    # Retrieves reviews with AT LEAST ONE aspect word OR AT LEAST ONE opinion word.

    aspect_terms, opinion_terms = parse_query(raw_query)
    
    mask_aspect = df['cleaned_review'].apply(
        lambda txt: any(term in str(txt) for term in aspect_terms)
    )
    mask_opinion = df['cleaned_review'].apply(
        lambda txt: any(term in str(txt) for term in opinion_terms)
    )
    return df[mask_aspect | mask_opinion]


if __name__ == "__main__":
    queries = [
        "audio quality:poor",
        "wifi signal:strong",
        "mouse button:click problem",
        "gps map:useful",
        "image quality:sharp",
    ]

    test_functions = {
        1: run_baseline_test1,
        2: run_baseline_test2,
        3: run_baseline_test3,
    }

    # --- Run Tests and Show Results ---
    print("-" * 50)
    print(f"{'Query':<30} | {'Test':<5} | {'# Retrieved':<10}")
    print("-" * 50)

    for query in queries:
        for test_num, test_func in test_functions.items():
            # Run the appropriate test function
            results_df = test_func(df, query)
            
            # Print the number of results found
            print(f"{query:<30} | {test_num:<5} | {len(results_df):<10}")
        print("-" * 50)

    print("\nAll baseline tests completed.")
