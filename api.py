import pickle
import re
import os
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
import nltk

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

nltk.download("stopwords")

app = FastAPI()

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# LOAD FILES
# =========================
log_model = pickle.load(open(os.path.join(BASE_DIR, "logistic_model.pkl"), "rb"))
nb_model = pickle.load(open(os.path.join(BASE_DIR, "naive_bayes_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

X_test = pickle.load(open(os.path.join(BASE_DIR, "X_test.pkl"), "rb"))
y_test = pickle.load(open(os.path.join(BASE_DIR, "y_test.pkl"), "rb"))

stop_words = set(stopwords.words("english"))

# =========================
# REQUEST MODEL
# =========================
class Tweet(BaseModel):
    text: str

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# =========================
# PREDICT
# =========================
@app.post("/predict")
def predict(tweet: Tweet):
    cleaned = clean_text(tweet.text)
    vec = vectorizer.transform([cleaned])

    lr_pred = log_model.predict(vec)[0]
    lr_conf = max(log_model.predict_proba(vec)[0])

    nb_pred = nb_model.predict(vec)[0]
    nb_conf = max(nb_model.predict_proba(vec)[0])

    return {
        "logistic_regression": {
            "sentiment": lr_pred,
            "confidence": float(lr_conf)
        },
        "naive_bayes": {
            "sentiment": nb_pred,
            "confidence": float(nb_conf)
        }
    }

# =========================
# METRICS (REAL)
# =========================
@app.get("/metrics")
def metrics():
    # Logistic Regression
    y_pred_lr = log_model.predict(X_test)

    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    lr_precision = precision_score(y_test, y_pred_lr, average="macro")
    lr_recall = recall_score(y_test, y_pred_lr, average="macro")
    lr_f1 = f1_score(y_test, y_pred_lr, average="macro")
    lr_cm = confusion_matrix(y_test, y_pred_lr).tolist()

    # Naive Bayes
    y_pred_nb = nb_model.predict(X_test)

    nb_accuracy = accuracy_score(y_test, y_pred_nb)
    nb_precision = precision_score(y_test, y_pred_nb, average="macro")
    nb_recall = recall_score(y_test, y_pred_nb, average="macro")
    nb_f1 = f1_score(y_test, y_pred_nb, average="macro")
    nb_cm = confusion_matrix(y_test, y_pred_nb).tolist()

    return {
        "logistic_regression": {
            "accuracy": float(lr_accuracy),
            "f1": float(lr_f1),
            "precision": float(lr_precision),
            "recall": float(lr_recall),
            "confusion_matrix": lr_cm
        },
        "naive_bayes": {
            "accuracy": float(nb_accuracy),
            "f1": float(nb_f1),
            "precision": float(nb_precision),
            "recall": float(nb_recall),
            "confusion_matrix": nb_cm
        }
    }