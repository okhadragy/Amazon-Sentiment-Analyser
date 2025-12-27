# Amazon Sentiment Analyser

This project performs sentiment analysis on **Amazon reviews** in the **Home & Kitchen** category.  
It includes data preprocessing, feature extraction, feature selection, model training, and evaluation.

---

## Dataset

We use the **Amazon Reviews 2023** dataset:  
[https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/)  

- **Category used:** `Home_and_Kitchen.jsonl`  
- Each review contains:
  - `rating` (1-5)
  - `title` (review title)
  - `text` (review body)  

> Make sure to download the dataset and place the file `Home_and_Kitchen.jsonl` in the project folder.

---

## Project Structure

amazon-sentiment-analyser/
├─ script.py # Main sentiment analysis script
├─ download_data.py # Downloads required NLTK resources
├─ Home_and_Kitchen.jsonl # Dataset (download manually)
├─ requirements.txt # Python dependencies
└─ README.md

---

## How to Run

1. Clone the repository or download the files.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install dependencies:

```bash
python download_data.py
```

4. Download the dataset from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) and place Home_and_Kitchen.jsonl in the project folder.

5. Run the main script:

```bash
python script.py
```

---

## Script Workflow

The script performs the following steps:

### 1. Data Preprocessing
- Lowercasing all text  
- Removing numbers and URLs  
- Tokenization  
- Stopword removal  
- Lemmatization  

### 2. Feature Extraction
- Bag of Words (BoW)  
- TF-IDF  
- Word2Vec embeddings  

### 3. Feature Selection
- Chi-Square for BoW  
- Mutual Information for BoW  
- ANOVA (F-test) for TF-IDF  

### 4. Classification Models
- Multinomial Naive Bayes (BoW + Chi-Square / Mutual Information)  
- Logistic Regression (TF-IDF + ANOVA)  
- Random Forest (Word2Vec)  

### 5. Evaluation
- Accuracy, precision, recall, F1-score  
- Specificity per class  
- Confusion matrix visualization with **Actual values on X-axis** and labels at the top
