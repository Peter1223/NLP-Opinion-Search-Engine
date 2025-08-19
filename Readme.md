# COSC 4397 - Natural Language Opinion Search Engine
**README**
======================================================================

This document provides step-by-step instructions to set up the environment and run the code for this project. The process involves a one-time data preparation step, followed by a single script to generate all required output files.

---
## **Part 1: Setup Instructions (3 Steps)**

Before running any code, please follow these setup steps to ensure the project directory is structured correctly.

### **Step 1: Final Folder and File Structure**

After unzipping the submission file, your main project folder should be organized as follows. You will need to add the `reviews.csv` file and the `opinion-lexicon-English` folder yourself.

Your_Project_Folder/
|
|-- Codes/                <-- (All .py scripts and .pkl models are here)
|   |-- preprocess_data.py
|   |-- generate_outputs.py
|   |-- evaluate.py
|   |-- baseline_tests.py
|   |-- method1.py
|   |-- method2_ml.py
|   |-- sentiment_utils.py
|   |-- logreg.pkl
|   |-- tfidf.pkl
|   +-- README.txt
|
|-- Outputs/              <-- (This will be created by the script)
|
|-- reviews.csv           <-- (You must add this file here)
|
+-- opinion-lexicon-English/  <-- (You must add this folder here)
|-- positive-words.txt
+-- negative-words.txt
 
### **Step 2: Add Required Data and Lexicon**

The dataset and lexicon are not included in the submission per project instructions. Please place them in the main project folder as shown above.

1.  **Dataset**: Place the `reviews.csv` file in the main project directory.
2.  **Lexicon**: Place the `opinion-lexicon-English` folder (containing `positive-words.txt` and `negative-words.txt`) in the main project directory.

### **Step 3: Install Required Libraries**

This project requires `pandas`, `scikit-learn`, and `nltk`. You can install them all with a single command in your terminal:

`pip install pandas scikit-learn nltk`

---
## **Part 2: How to Run the Code (2 Steps)**

After completing the setup, please follow these two steps to generate all required output files.

### **Step 1: Prepare the Data (Run Once)**

First, you need to run the preprocessing script to clean the raw review data. This will create the `cleaned_reviews.csv` file that the main script needs.

Navigate to your main project folder in the terminal and run:

`python Codes/preprocess_data.py`

### **Step 2: Generate All Output Files**

Once the `cleaned_reviews.csv` file has been created, you can generate all 20 required output files for the submission by running the following command from the main project folder:

`python Codes/generate_outputs.py`

This script will automatically create the `Outputs/` folder with its two subfolders (`baseline_model_outputs` and `adv_model_outputs`) and save all 20 `.txt` files to their correct locations.

---
## **(Optional) How to Reproduce Report Evaluation**

To see the comparison table of retrieved document counts that was used in the final report, you can run the evaluation script:

`python Codes/evaluate.py`

This will print a summary table to the console showing the number of results retrieved by each model for each query.
