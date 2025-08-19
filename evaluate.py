import os
import pandas as pd
from baseline_tests import run_baseline_test2
from method1 import search_with_rating
from method2_ml import search_with_ml

# Create a directory to store detailed results for manual review
EVAL_DIR = "evaluation_results"
if not os.path.exists(EVAL_DIR):
    os.makedirs(EVAL_DIR)

df = pd.read_csv("cleaned_reviews.csv")

test_queries = [
    "audio quality:poor",
    "wifi signal:strong",
    "mouse button:click problem",
    "gps map:useful",
    "image quality:sharp",
]

print("-" * 60)
print(f"{'Query':<30} | {'# Ret. Baseline':<15} | {'# Ret. M1':<10} | {'# Ret. M2':<10}")
print("-" * 60)

for q in test_queries:
    models = {
        # Use run_baseline_test2 as the main "Baseline" for report table
        "Baseline": run_baseline_test2(df, q), 
        "M1": search_with_rating(df, q),
        "M2": search_with_ml(q, df)
    }

    # Save results to files
    aspect_part = q.split(":")[0].strip().replace(" ", "_")
    for model_name, results_df in models.items():
        if not results_df.empty:
            filename = os.path.join(EVAL_DIR, f"{aspect_part}_{model_name}_results.csv")
            # Save review_id and the original text to make reviewing easier
            results_df[['review_id', 'review_text']].to_csv(filename, index=False)

    # Print the summary table
    num_base = len(models["Baseline"])
    num_m1 = len(models["M1"])
    num_m2 = len(models["M2"])
    print(f"{q:<30} | {num_base:<15} | {num_m1:<10} | {num_m2:<10}")

print("-" * 60)
print(f"\nSaved detailed results for all models in the '{EVAL_DIR}' folder.")