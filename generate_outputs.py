import os
import pandas as pd

# Import the search functions from your other scripts
from baseline_tests import run_baseline_test1, run_baseline_test2, run_baseline_test3
from method2_ml import search_with_ml

# Define output directories
BASELINE_OUTPUT_DIR = "Outputs/baseline_model_outputs"
ADVANCED_OUTPUT_DIR = "Outputs/adv_model_outputs"

# Create directories if they don't exist
os.makedirs(BASELINE_OUTPUT_DIR, exist_ok=True)
os.makedirs(ADVANCED_OUTPUT_DIR, exist_ok=True)

# The 5 official evaluation queries
test_queries = [
    "audio quality: poor",
    "wifi signal: strong",
    "mouse button: click problem",
    "gps map: useful",
    "image quality: sharp",
]

# Load Data 
try:
    df = pd.read_csv("cleaned_reviews.csv")
except FileNotFoundError:
    print("Error: 'cleaned_reviews.csv' not found. Please ensure it is in the correct directory.")
    exit()

# 1. Generate Baseline Model Outputs 
print("-" * 50)
print(f"Generating 15 baseline output files in '{BASELINE_OUTPUT_DIR}'...")
print("-" * 50)

baseline_test_functions = {
    1: run_baseline_test1,
    2: run_baseline_test2,
    3: run_baseline_test3,
}

for query in test_queries:
    for test_num, test_func in baseline_test_functions.items():
        print(f"Running Baseline Test {test_num} for query: '{query}'...")
        
        results_df = test_func(df, query)
        
        base_filename = query.split(":")[0].replace(" ", "_")
        output_filename = f"{base_filename}_test{test_num}.txt"
        output_path = os.path.join(BASELINE_OUTPUT_DIR, output_filename)
        
        if 'review_id' in results_df.columns:
            results_df['review_id'].to_csv(output_path, header=False, index=False)
            print(f"  -> Saved {len(results_df)} results to {output_path}\n")
        else:
            print(f"  -> Error: 'review_id' column not found for baseline query '{query}'.")


# 2. Generate Advanced Model (M2) Outputs
print("-" * 50)
print(f"Generating 5 advanced model output files in '{ADVANCED_OUTPUT_DIR}'...")
print("-" * 50)

for q in test_queries:
    print(f"Running Method 2 for query: '{q}'...")
    results = search_with_ml(q, df)

    aspect_part = q.split(":")[0].strip().replace(" ", "_")
    output_filename = os.path.join(ADVANCED_OUTPUT_DIR, f"{aspect_part}.txt")

    if not results.empty:
        if 'review_id' in results.columns:
            results['review_id'].to_csv(output_filename, header=False, index=False)
            print(f"  -> Saved {len(results)} results to {output_filename}\n")
        else:
             print(f"  -> Error: 'review_id' column not found in results for M2 query '{q}'.\n")
    else:
        with open(output_filename, 'w') as f:
            pass # Creates an empty file
        print(f"  -> No results for '{q}'. Created empty file: {output_filename}\n")


print("-" * 50)
print("Done! All 20 output files have been generated.")
