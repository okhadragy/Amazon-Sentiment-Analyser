# -------- imports ----------
import re, json, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from gensim.models import Word2Vec
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Hyperparameters ----------
chunk_size = 100000                 # size of used data (train + test)
bow_max_features = 5000             # number of max features for BoW
tfidf_max_features = 5000           # number of max features for TF-IDF
w2v_max_features = 100              # number of max features for Word2Vec
w2v_window_size = 5                 # number of neighbours around target word 
w2v_min_count = 2                   # min frequency of word to be included in the training corpus.
w2v_workers = 4                     # number of CPU threads to use for Word2Vec training
bow_chi_k = 2000                    # number of top features to select for BoW using Chi-Square
bow_mi_k = 2000                     # number of top features to select for BoW using mutual information
tfidf_anova_k = 2000                # number of top features to select for TF-IDF using ANOVA
nb_alpha = 1.0                      # [Multinomial Naive Bayes] Laplace smoothing parameter (default 1.0)
lr_max_iter = 1000                  # [Logistic Regression] max number of iterations for convergence
lr_solver = 'lbfgs'                 # [Logistic Regression] solver that supports multinomial loss
lr_C = 1.0                          # [Logistic Regression] regularization strength (inverse, default 1.0)
lr_l1_ration = 0                    # [Logistic Regression] regularization type
rf_n_estimators = 200               # [Random Forest] number of trees
rf_max_depth = None                 # [Random Forest] max depth of each tree
rf_min_samples_split = 2            # [Random Forest] min samples required to split a node
rf_min_samples_leaf = 1             # [Random Forest] min samples at a leaf node
rf_max_features = 'sqrt'            # [Random Forest] max features to consider at split

# ----- Loading Data --------
file = "Home_and_Kitchen.jsonl"

def read_data_chunk(file_path, chunk_size=100000):
    chunk = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            review = json.loads(line)
            chunk.append({
                "rating": review.get("rating"),
                "title": review.get("title"),
                "text": review.get("text")
            })
            if i >= chunk_size: 
                return chunk
    
df = pd.DataFrame(read_data_chunk(file, chunk_size))

# ----- Preprocessing --------

# Drop missing text
df.dropna(subset=['text'], inplace=True)

# Map sentiment
def map_sentiment(rating):
    if rating <= 2:
        return 0 # negative
    elif rating == 3:
        return 1 # neutral
    else:
        return 2 # positive

df['sentiment'] = df['rating'].apply(map_sentiment)

# Split data
X = df[['title', 'text']]
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# lower, remove numbers and urls, split to tokens, stop words, Lemmatize
SAFE_STOPWORDS = {
    'a','an','the',
    'and','or','but',
    'if','while','as',
    'of','at','by','for','from',
    'with','about','into','through',
    'above','below','up','down','out','off','over','under',
    'once','again',
    'each','few','more','most','some',
    'such','only','own','same',
    'so','than','too','very'
}

PRONOUN_STOPWORDS = {
    'i','me','my','myself',
    'we','our','ours','ourselves',
    'you','your','yours','yourself','yourselves',
    'he','him','his','himself',
    'she','her','hers','herself',
    'they','them','their','theirs','themselves',
    'it','its','itself'
}

AUX_VERBS = {
    'am','is','are','was','were','be','been','being',
    'have','has','had','having',
    'do','does','did','doing',
    'will','would','shall','should','can','could','may','might','must'
}

CUSTOM_STOPWORDS = (
    SAFE_STOPWORDS
    | PRONOUN_STOPWORDS
    | AUX_VERBS
)

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|\d+", "", text)
    tokens = text.split()
    pos_tags = nltk.pos_tag(tokens)
    processed_tokens = []
    for word, tag in pos_tags:
        if word not in CUSTOM_STOPWORDS:
            wn_tag = get_wordnet_pos(tag)
            processed_tokens.append(lemmatizer.lemmatize(word, pos=wn_tag))
    return " ".join(processed_tokens)


X_train_text = (X_train['title'] + " " + X_train['text']).apply(preprocess)
X_test_text  = (X_test['title'] + " " + X_test['text']).apply(preprocess)

# ---------- Feature Extraction --------------

# box of words (BoW)
bow = CountVectorizer(max_features=bow_max_features)
X_train_bow = bow.fit_transform(X_train_text)
X_test_bow  = bow.transform(X_test_text)

# TF-IDF
tfidf = TfidfVectorizer(max_features=tfidf_max_features)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf  = tfidf.transform(X_test_text)

# word to verctor (Word2Vec)
sentences = [text.split() for text in X_train_text]
w2v = Word2Vec(sentences, vector_size=w2v_max_features, window=w2v_window_size, min_count=w2v_min_count, workers=w2v_workers)
def sentence_vector(sentence):
    words = sentence.split()
    vectors = [w2v.wv[w] for w in words if w in w2v.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_max_features)
X_train_w2v = np.vstack(X_train_text.apply(sentence_vector))
X_test_w2v  = np.vstack(X_test_text.apply(sentence_vector))


# ----------- Feature Selection ---------------

# Chi-Square for BoW
selector_bow = SelectKBest(score_func=chi2, k=bow_chi_k)
X_train_bow_sel_chi = selector_bow.fit_transform(X_train_bow, y_train)
X_test_bow_sel_chi  = selector_bow.transform(X_test_bow)

# Mutual Information for BoW
selector_bow = SelectKBest(score_func=mutual_info_classif, k=bow_mi_k)
X_train_bow_sel_mi = selector_bow.fit_transform(X_train_bow, y_train)
X_test_bow_sel_mi  = selector_bow.transform(X_test_bow)

# ANOVA for TF-IDF
selector_tfidf = SelectKBest(score_func=f_classif, k=tfidf_anova_k)
X_train_tfidf_sel_anova = selector_tfidf.fit_transform(X_train_tfidf, y_train)
X_test_tfidf_sel_anova  = selector_tfidf.transform(X_test_tfidf)


# ---------- Train models ------------

# For BoW + Chi-square + MultinomialNB
clf_bow_chi = MultinomialNB(alpha=nb_alpha)
clf_bow_chi.fit(X_train_bow_sel_chi, y_train)

# For BoW + Mutual Information + MultinomialNB
clf_bow_mi = MultinomialNB(alpha=nb_alpha)
clf_bow_mi.fit(X_train_bow_sel_mi, y_train)

# For TF-IDF + ANOVA + Logistic Regression
clf_tfidf = LogisticRegression(max_iter=lr_max_iter, solver=lr_solver, C=lr_C, l1_ratio=lr_l1_ration)
clf_tfidf.fit(X_train_tfidf_sel_anova, y_train)

# For Word2Vec + Random Forest
clf_w2v = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, min_samples_split=rf_min_samples_split, min_samples_leaf=rf_min_samples_leaf, max_features=rf_max_features, random_state=42)
clf_w2v.fit(X_train_w2v, y_train)


# --------- Evaluate Models ----------

# Evaluation function
def evaluate_model(y_true, y_pred, class_names=['negative','neutral','positive']):
    print("\nAccuracy:", accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred) # Confusion matrix
    
    # Specificity per class: TN / (TN + FP)
    # For multi-class, specificity is calculated per class
    num_classes = len(class_names)
    specificity = []
    for i in range(num_classes):
        TP = cm[i,i]
        FP = cm[:,i].sum() - TP
        FN = cm[i,:].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        specificity.append(TN / (TN + FP))
        
    for i, s in enumerate(specificity):
        print(f"Specificity for class {class_names[i]}: {s:.3f}")
        
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
    
    # Reversed Confusion matrix
    cm_rev = cm.T
    print("Confusion Matrix:\n", cm_rev)
    
    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    ax = sns.heatmap(cm_rev, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    plt.show()

    

# For BoW + Chi-square + MultinomialNB
preds_bow = clf_bow_chi.predict(X_test_bow_sel_chi)
print("\n---------- BoW + Chi-square + MultinomialNB -----------")
evaluate_model(y_test, preds_bow)

# For BoW + Mutual Information + MultinomialNB
preds_bow = clf_bow_mi.predict(X_test_bow_sel_mi)
print("\n---------- BoW + Mutual Information + MultinomialNB -----------")
evaluate_model(y_test, preds_bow)

# For TF-IDF + ANOVA + Logistic Regression
preds_tfidf = clf_tfidf.predict(X_test_tfidf_sel_anova)
print("\n---------- TF-IDF + ANOVA + Logistic Regression -----------")
evaluate_model(y_test, preds_tfidf)

# For Word2Vec + Random Forest
preds_w2v = clf_w2v.predict(X_test_w2v)
print("\n---------- Word2Vec + Random Forest -----------")
evaluate_model(y_test, preds_w2v)