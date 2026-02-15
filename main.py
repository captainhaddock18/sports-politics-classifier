# ------------------------------------------------------------
# SPORTS OR POLITICS CLASSIFIER
# Using BBC RSS-style dataset
# ------------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ------------------------------------------------------------
# STEP 1: Load Dataset
# ------------------------------------------------------------

# Replace with your file name
df = pd.read_csv("bbc_news.csv")

print("Dataset shape:", df.shape)
print(df.head())
df = df.head(5000)


# ------------------------------------------------------------
# STEP 2: Create Labels from URL
# ------------------------------------------------------------
# We generate labels based on link patterns.

def generate_label(url):
    url = str(url).lower()

    if "sport" in url:
        return "sport"

    # Everything else treated as politics/news
    elif any(keyword in url for keyword in
             ["politics", "uk", "world", "europe", "business"]):
        return "politics"

    else:
        return None


df["label"] = df["link"].apply(generate_label)

# Remove rows without label
df = df[df["label"].notnull()]

print("Label distribution:")
print(df["label"].value_counts())


# ------------------------------------------------------------
# STEP 3: Feature Selection
# ------------------------------------------------------------
# We use 'description' column as text

X = df["description"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# STEP 4: FEATURE REPRESENTATION
# ------------------------------------------------------------

# 1️⃣ Bag of Words
bow_vectorizer = CountVectorizer(stop_words="english")
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# 2️⃣ TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 3️⃣ TF-IDF with Bigrams
tfidf_bigram = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)
X_train_bigram = tfidf_bigram.fit_transform(X_train)
X_test_bigram = tfidf_bigram.transform(X_test)


# ------------------------------------------------------------
# STEP 5: MODEL TRAINING AND EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n====================================")
    print("Model:", model_name)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# ------------------------------------------------------------
# STEP 6: APPLY MODELS
# ------------------------------------------------------------

# 1️⃣ Naive Bayes
nb = MultinomialNB()

evaluate_model(nb, X_train_bow, X_test_bow, y_train, y_test, "Naive Bayes (BoW)")
evaluate_model(nb, X_train_tfidf, X_test_tfidf, y_train, y_test, "Naive Bayes (TF-IDF)")

# 2️⃣ Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced')

evaluate_model(lr, X_train_tfidf, X_test_tfidf, y_train, y_test, "Logistic Regression (TF-IDF)")
evaluate_model(lr, X_train_bigram, X_test_bigram, y_train, y_test, "Logistic Regression (TF-IDF + Bigrams)")

# 3️⃣ Support Vector Machine
svm = LinearSVC(class_weight='balanced')

evaluate_model(svm, X_train_tfidf, X_test_tfidf, y_train, y_test, "SVM (TF-IDF)")
evaluate_model(svm, X_train_bigram, X_test_bigram, y_train, y_test, "SVM (TF-IDF + Bigrams)")
