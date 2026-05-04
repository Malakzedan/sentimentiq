# ◈ SentimentIQ — Twitter Sentiment Analysis Dashboard

A real-time NLP sentiment analysis dashboard built with **Streamlit** and **FastAPI**, using machine learning to classify airline tweets as **positive**, **neutral**, or **negative**.

---

## ✨ Features

- **Two ML Models** — Logistic Regression and Naive Bayes with GridSearchCV hyperparameter tuning
- **Live Prediction** — Analyze any text input in real-time with both models
- **Confidence Scores** — View prediction confidence for both models
- **Interactive Dashboard** — Model comparison charts, confusion matrices, and metric cards
- **Dark/Light Mode** — Toggle between themes

---

## 🛠 Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| Frontend    | Streamlit, Matplotlib               |
| Backend     | FastAPI, Uvicorn                    |
| ML Models   | Scikit-learn (LogisticRegression, MultinomialNB) |
| NLP         | TF-IDF Vectorization, NLTK          |
| Dataset     | [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (14,640 tweets) |

---

## 📊 Model Performance

| Metric    | Logistic Regression | Naive Bayes |
|-----------|:-------------------:|:-----------:|
| Accuracy  | 0.785               | 0.758       |
| Precision | 0.760 (macro)       | 0.763 (macro) |
| Recall    | 0.657 (macro)       | 0.577 (macro) |
| F1 Score  | 0.692 (macro)       | 0.622 (macro) |

---

## 📁 Project Structure

```
├── api.py                  # FastAPI backend (predictions + metrics)
├── app.py                  # Streamlit frontend dashboard
├── train.py                # Model training script
├── Tweets.csv              # Dataset
├── logistic_model.pkl      # Trained Logistic Regression model
├── naive_bayes_model.pkl   # Trained Naive Bayes model
├── vectorizer.pkl          # TF-IDF vectorizer
├── X_test.pkl              # Test features
├── y_test.pkl              # Test labels
├── requirements.txt        # Python dependencies
├── start.sh                # Startup script (for deployment)
└── .streamlit/
    └── config.toml         # Streamlit server config
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Malakzedan/sentimentiq.git
   cd sentimentiq
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux/Mac
   .venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Retrain the models**
   ```bash
   python train.py
   ```

### Running the App

Open **two terminals** in the project folder:

**Terminal 1 — Start the API:**
```bash
uvicorn api:app --reload --port 8000
```

**Terminal 2 — Start the dashboard:**
```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## 📸 Screenshots

![img.png](img.png)
![img_1.png](img_1.png)
![img_2.png](img_2.png)
![img_3.png](img_3.png)
![img_4.png](img_4.png)

---

## 👩‍💻 Author

**Malak Zedan**

---

