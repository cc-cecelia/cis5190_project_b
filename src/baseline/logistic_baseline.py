import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss

input_dir = "./data/processed/processed_data.csv"
log_dir = "./src/models/logs"


def main(df, tf=True, max_iter=100):
    X = df["title"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    y_train = y_train.apply(lambda x: 1 if x == "fox" else 0)
    y_test = y_test.apply(lambda x: 1 if x == "fox" else 0)

    if tf:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
        feature_type = "baseline_tfidf"
    else:
        vectorizer = CountVectorizer(stop_words="english", max_features=100)
        feature_type = "baseline_tf"

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1,
        warm_start=True,
        solver="lbfgs"
    )

    train_losses = []

    for i in range(max_iter):
        model.fit(X_train_vec, y_train)
        prob = model.predict_proba(X_train_vec)
        loss = log_loss(y_train, prob)
        train_losses.append(loss)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    os.makedirs(log_dir, exist_ok=True)
    print(log_dir)
    log_path = os.path.join(log_dir, f"logistic_{feature_type}_loss.json")

    with open(log_path, "w") as f:
        json.dump(
            {
                "feature_type": feature_type,
                "max_iter": max_iter,
                "train_loss": train_losses,
            },
            f,
            indent=2,
        )

    return accuracy


if __name__ == "__main__":
    df = pd.read_csv(input_dir)

    print("raw data summary:")
    print(df["label"].value_counts())
    print(df["label"].value_counts(normalize=True))

    print("\nbaseline + TF-IDF")
    main(df, tf=True)

    print("\nbaseline + TF")
    main(df, tf=False)
