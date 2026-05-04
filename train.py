import pandas as pd
import re
import nltk
import pickle
import os

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# CLEAN TEXT
# =========================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(os.path.join(BASE_DIR, "Tweets.csv"))
df = df[['text', 'airline_sentiment']]
df.columns = ['text', 'sentiment']

df['clean_text'] = df['text'].apply(clean_text)

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SAVE TEST DATA
pickle.dump(X_test, open(os.path.join(BASE_DIR, "X_test.pkl"), "wb"))
pickle.dump(y_test, open(os.path.join(BASE_DIR, "y_test.pkl"), "wb"))

# =========================
# MODELS
# =========================
lr = LogisticRegression(max_iter=200)
nb = MultinomialNB()

# =========================
# GRID SEARCH - LR
# =========================
lr_params = {
    "C": [0.1, 1, 10],
    "solver": ["lbfgs", "saga"]
}

grid_lr = GridSearchCV(lr, lr_params, cv=3, n_jobs=-1)
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_

# =========================
# GRID SEARCH - NB
# =========================
nb_params = {
    "alpha": [0.5, 1.0, 1.5]
}

grid_nb = GridSearchCV(nb, nb_params, cv=3)
grid_nb.fit(X_train, y_train)
best_nb = grid_nb.best_estimator_

# =========================
# EVALUATION
# =========================
lr_pred = best_lr.predict(X_test)
nb_pred = best_nb.predict(X_test)

print("\n===== ACCURACY =====")
print("Logistic Regression:", accuracy_score(y_test, lr_pred))
print("Naive Bayes:", accuracy_score(y_test, nb_pred))

print("\n===== LR REPORT =====")
print(classification_report(y_test, lr_pred))

print("\n===== NB REPORT =====")
print(classification_report(y_test, nb_pred))

print("\n===== LR CONFUSION MATRIX =====")
print(confusion_matrix(y_test, lr_pred))

print("\n===== NB CONFUSION MATRIX =====")
print(confusion_matrix(y_test, nb_pred))

# =========================
# SAVE MODELS
# =========================
pickle.dump(best_lr, open(os.path.join(BASE_DIR, "logistic_model.pkl"), "wb"))
pickle.dump(best_nb, open(os.path.join(BASE_DIR, "naive_bayes_model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(BASE_DIR, "vectorizer.pkl"), "wb"))

print("\nModels and test data saved successfully!")