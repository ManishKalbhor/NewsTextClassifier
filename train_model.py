import os
import re
import json
import joblib
import nltk
import pandas as pd

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

os.makedirs("model", exist_ok=True)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = [
        word
        for word in text.split()
        if word not in STOP_WORDS
    ]

    return " ".join(words)


# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading dataset...")

fake = pd.read_csv(
    "dataset/Fake.csv"
)

real = pd.read_csv(
    "dataset/True.csv"
)

print("Fake articles:", len(fake))
print("Real articles:", len(real))

print("\nFake columns:")
print(fake.columns.tolist())

print("\nReal columns:")
print(real.columns.tolist())


# =========================================================
# LABEL DATA
# =========================================================

fake["label"] = 0
real["label"] = 1


# =========================================================
# COMBINE DATA
# =========================================================

data = pd.concat(
    [fake, real],
    ignore_index=True
)


# =========================================================
# HEADLINE + ARTICLE
# =========================================================

data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

data["content"] = (
    data["title"] + " " + data["text"]
)


# =========================================================
# CLEAN DATA
# =========================================================

data["content"] = data["content"].apply(
    clean_text
)

data = data[
    data["content"].str.len() > 0
]

data = data.drop_duplicates()


X = data["content"]
y = data["label"]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =========================================================
# TF-IDF
# =========================================================

print("\nCreating TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


# =========================================================
# MODELS
# =========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Multinomial Naive Bayes":
        MultinomialNB(),

    "Linear SVM":
        LinearSVC()
}


results = {}

trained_models = {}


# =========================================================
# TRAIN MODELS
# =========================================================

for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    predictions = model.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Fake",
            "Real"
        ],
        output_dict=True
    )

    results[name] = {

        "accuracy":
            float(accuracy),

        "confusion_matrix":
            cm.tolist(),

        "classification_report":
            report

    }

    trained_models[name] = model

    print(
        f"{name} Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


# =========================================================
# FIND BEST MODEL
# =========================================================

best_model_name = max(
    results,
    key=lambda x:
    results[x]["accuracy"]
)

best_model = trained_models[
    best_model_name
]

print("\n================================")
print("MODEL COMPARISON")
print("================================")

for name, result in results.items():

    print(
        f"{name}: "
        f"{result['accuracy'] * 100:.2f}%"
    )

print(
    "\nBest Model:",
    best_model_name
)


# =========================================================
# SAVE BEST MODEL
# =========================================================

joblib.dump(
    best_model,
    "model/fake_news_model.pkl"
)


# =========================================================
# SAVE VECTORIZER
# =========================================================

joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)


# =========================================================
# SAVE ALL MODELS
# =========================================================

for name, model in trained_models.items():

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "")
        + ".pkl"
    )

    joblib.dump(
        model,
        f"model/{filename}"
    )


# =========================================================
# SAVE METRICS
# =========================================================

metrics = {

    "best_model":
        best_model_name,

    "results":
        results,

    "dataset":
    {
        "fake_articles":
            len(fake),

        "real_articles":
            len(real),

        "total_articles":
            len(data),

        "training_articles":
            len(X_train),

        "testing_articles":
            len(X_test)
    }
}


with open(
    "model/metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# =========================================================
# SAVE ACCURACY
# =========================================================

best_accuracy = (
    results[
        best_model_name
    ]["accuracy"] * 100
)

with open(
    "model/accuracy.txt",
    "w"
) as file:

    file.write(
        f"{best_accuracy:.2f}"
    )


print("\n================================")
print("TRAINING COMPLETE")
print("================================")

print(
    "Best Model:",
    best_model_name
)

print(
    "Best Accuracy:",
    f"{best_accuracy:.2f}%"
)

print("\nFiles created:")

print(
    "model/fake_news_model.pkl"
)

print(
    "model/tfidf_vectorizer.pkl"
)

print(
    "model/metrics.json"
)

print(
    "model/accuracy.txt"
)