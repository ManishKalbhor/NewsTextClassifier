# FakeNewsTextClassifier — Project Report

**Report date:** 30 October 2026

**Project type:** Text classification web application

**Interface:** Streamlit

**Python target:** 3.11 or 3.12

## 1. Project summary

FakeNewsTextClassifier is a machine learning application that classifies a supplied news headline and/or article as **Fake** or **Real**. It uses a TF-IDF representation of the text and a scikit-learn classifier. The Streamlit interface accepts text, displays the model label and influential terms, provides model evaluation views, and keeps a temporary history of analyses.

The output is a prediction from learned text patterns. It does not verify claims, sources, images, or events and should not be treated as a fact-checking result.

## 2. Objectives

- Accept a headline, article, or both from a user.
- Apply the same general text-cleaning steps used in training.
- Predict a binary label: Fake (`0`) or Real (`1`).
- Explain some influential words and phrases for the selected linear model.
- Show dataset and model evaluation information in a web interface.

## 3. Dataset

The training script reads two local CSV files:

- `dataset/Fake.csv`
- `dataset/True.csv`

Both files have the columns `title`, `text`, `subject`, and `date`. The script assigns label `0` to Fake rows and `1` to True rows. It combines the title and article body into one text field; `subject` and `date` are not used as model features.

| Dataset count | Records |
|---|---:|
| Fake source rows | 23,481 |
| Real source rows | 21,417 |
| Raw source rows combined | 44,898 |
| Records used after empty-text filtering and full-row deduplication | 44,680 |
| Training records | 35,744 |
| Held-out test records | 8,936 |

The CSV files are excluded from Git by `.gitignore`. A local copy is needed to retrain the model; they are not needed to run the app when the model artifacts are present.

## 4. Text preparation and feature extraction

The training and app code perform these steps:

1. Join the headline and article body.
2. Convert text to lowercase.
3. Remove URLs.
4. Replace non-English-letter characters with spaces.
5. Normalize repeated whitespace.
6. Remove English NLTK stopwords.
7. Convert the cleaned text into TF-IDF features.

The TF-IDF vectorizer uses a maximum of 50,000 features, unigrams and bigrams (`ngram_range=(1, 2)`), `min_df=2`, and sublinear term frequency. It is fit on the training portion and then applied to the test portion.

## 5. Training and evaluation method

The script uses a stratified 80/20 train/test split with `random_state=42`. It trains three candidate models and records accuracy, a confusion matrix, and a classification report for each:

- Logistic Regression with `max_iter=1000`
- Multinomial Naive Bayes
- Linear Support Vector Machine (`LinearSVC`)

The model with the highest recorded accuracy is saved as `model/fake_news_model.pkl`. The TF-IDF vectorizer is saved separately. Detailed metrics are written to `model/metrics.json`; the best accuracy is also written to `model/accuracy.txt`.

## 6. Reported model results

Results below are from the project's existing `model/metrics.json` artifact. Percentages are rounded to two decimal places.

| Model | Accuracy | Macro F1 | Errors in 8,936 test records |
|---|---:|---:|---:|
| Linear SVM (selected) | 99.87% | 99.87% | 12 |
| Logistic Regression | 99.45% | 99.45% | 49 |
| Multinomial Naive Bayes | 96.31% | 96.30% | 330 |

### Linear SVM class results

| Class | Precision | Recall | F1 score | Test support |
|---|---:|---:|---:|---:|
| Fake | 99.85% | 99.89% | 99.87% | 4,694 |
| Real | 99.88% | 99.83% | 99.86% | 4,242 |

Linear SVM confusion matrix (rows are actual labels; columns are predicted labels, ordered Fake then Real):

| Actual / Predicted | Fake | Real |
|---|---:|---:|
| Fake | 4,689 | 5 |
| Real | 7 | 4,235 |

## 7. Application features

- **Detector:** headline and article text inputs, prediction label, model score or decision margin, influential terms, and direction of influence. Inputs are length-limited, and the app warns if none of the entered terms are in the model vocabulary.
- **Overview:** test accuracy, selected model, train/test counts, and raw source row counts.
- **Performance:** model comparison, accuracy and macro F1 chart, confusion matrix, and class-level report.
- **History:** session-scoped recent analyses, CSV download, and clear-history action. The app limits history to the latest 100 entries.
- **Interface design:** dark slate theme with teal and coral accents, responsive Streamlit components, hover effects, field focus highlighting, and reduced-motion support.

The app loads artifacts from the `model/` directory relative to `app.py` and caches the model and vectorizer in Streamlit. History is held in Streamlit session state; it is not stored in a database.

## 8. Technology and project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface, input cleaning, prediction, explanations, dashboards, and history |
| `train_model.py` | Dataset preparation, TF-IDF fitting, model training, evaluation, and artifact export |
| `model/fake_news_model.pkl` | Selected classifier used by the app |
| `model/tfidf_vectorizer.pkl` | Fitted text vectorizer used by the app |
| `model/metrics.json` | Evaluation and dataset metrics displayed in the app |
| `model/accuracy.txt` | Best model accuracy as a percentage string |
| `requirements.txt` | Python package dependencies |
| `.streamlit/config.toml` | Dark theme, application colors, and Streamlit server setting |
| `startups.sh` | Starts Streamlit on port 8000 and binds to `0.0.0.0` |
| `.deployment` | Deployment build configuration |
| `README.md` | Local run, artifact, training-data, and deployment instructions |

Dependencies listed in `requirements.txt`: Streamlit, pandas, NumPy, scikit-learn 1.9.1, NLTK, joblib, and Matplotlib. Scikit-learn 1.9.1 is pinned to match the version marker found in the saved classifier; the other packages are not pinned. Seaborn was removed because the app and training script do not use it. The README targets Python 3.11 or 3.12; scikit-learn 1.9 requires Python 3.11 or newer ([official installation guide](https://scikit-learn.org/stable/install)).

## 9. Deployment and operation

The README describes local execution and Streamlit Community Cloud deployment. The repository also has a startup script that runs Streamlit on port `8000`, which is suitable for a hosted service that supplies that port. The three required runtime artifacts are the selected model, vectorizer, and metrics JSON; training CSVs are not needed for inference.

To retrain, place both CSV files in `dataset/`, install the dependencies, and run `python train_model.py` from the project root. Training downloads the NLTK English stopword corpus. The generated model artifacts should be deployed together with the dependency versions used to create them.

## 10. Limitations and risks

1. **Generalization:** The reported score is from a random split of the same dataset. It does not measure performance on newer stories, unseen publishers, or new topics.
2. **Potential duplicate leakage:** `drop_duplicates()` removes exact full-row duplicates, but rows with the same cleaned article text and different metadata can remain. Similar stories may also cross the random train/test split.
3. **Model selection:** The best model is selected by accuracy on the reported test set. A separate validation set would provide a cleaner final estimate after selecting a model.
4. **Score interpretation:** Linear SVM provides a decision margin, not a calibrated probability. The app labels it accordingly. Probability outputs from other classifiers may also need calibration before they are interpreted as likelihoods.
5. **Explanation limits:** Influential terms show how a linear model's text features contributed to a label. They are not evidence that the story's claims are true or false.
6. **Preprocessing mismatch:** If the NLTK stopword corpus is missing at app startup, `app.py` uses a smaller fallback list. This can make prediction preprocessing differ from training.
7. **Reproducibility:** Scikit-learn is pinned to the saved classifier's version, but the remaining dependencies and Python version are not fully locked. Pickle artifacts can depend on compatible scikit-learn and Python package versions.
8. **History and privacy:** History is temporary and session-scoped. No persistent database or cross-session history is configured.

## 11. Recommended next steps

1. Evaluate on a time-based holdout and a source-separated holdout; ideally add an external dataset.
2. Deduplicate on normalized text and investigate conflicting labels before splitting. Consider grouping related stories so a story cannot appear in both train and test.
3. Use separate train, validation, and final test sets. Select models on validation data and reserve the final test set for one final evaluation.
4. Make training and inference preprocessing identical by packaging the stopword list or storing the cleaner and vectorizer in a single inference pipeline.
5. Record Python and dependency versions with each model release; lock a tested set of dependencies.
6. If probabilities are needed, calibrate the selected classifier on validation data and report calibration metrics.
7. Add a small repeatable smoke check for artifact loading, empty input, headline-only input, and article input.

## 12. Conclusion

The project demonstrates an end-to-end NLP classification workflow: CSV preparation, text cleaning, TF-IDF feature generation, comparison of classical classifiers, model export, and an interactive Streamlit app. Its current measured test performance is high on the existing random split, but independent and time-aware evaluation is necessary before treating those metrics as evidence of real-world reliability.
