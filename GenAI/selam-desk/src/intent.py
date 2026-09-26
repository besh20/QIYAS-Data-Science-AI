"""Classical intent classifier: TF-IDF + Logistic Regression.
This is the piece that ties the project back to the classical ML you covered
in the course, and it's what you show side-by-side with the LLM router in
your evaluation section.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "questions.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "intent_model.joblib")


def train():
    df = pd.read_csv(DATA_PATH)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(df["text"], df["intent"])
    joblib.dump(pipe, MODEL_PATH)
    return pipe


def load_or_train():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train()


def predict(text: str, pipe=None):
    pipe = pipe or load_or_train()
    intent = pipe.predict([text])[0]
    proba = max(pipe.predict_proba([text])[0])
    return intent, float(proba)


if __name__ == "__main__":
    pipe = train()
    for t in ["registration meche new?", "how much is the fee?", "I want to talk to someone"]:
        print(t, "->", predict(t, pipe))
