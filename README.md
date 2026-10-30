# FakeNewsTextClassifier

A Streamlit app that screens a headline and article with a trained TF-IDF classifier, shows influential terms, and reports model evaluation metrics. The result is a model label, not a fact check or a determination of truth.

## Run locally

Use Python 3.11 or 3.12, then run these commands from the project folder. The saved model uses scikit-learn 1.9.1, which requires Python 3.11 or newer. See the [scikit-learn installation guide](https://scikit-learn.org/stable/install) for current platform requirements.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app needs these artifacts in `model/`:

- `fake_news_model.pkl`
- `tfidf_vectorizer.pkl`
- `metrics.json`

The saved classifier was created with scikit-learn 1.9.1; `requirements.txt` pins that version so it matches the pickle artifact. If you retrain the model, run `python train_model.py` and deploy the updated model, vectorizer, and metrics together. Record the full Python and package versions used for each release because pickle files depend on compatible libraries.

### Training data

`train_model.py` expects `dataset/Fake.csv` and `dataset/True.csv`. These files are excluded from Git by `.gitignore`, so add the dataset locally before training. The app can run without the training CSVs as long as the three model artifacts above are present.

## Known limitations and next improvements

- The reported accuracy comes from a random held-out split of the same dataset. It does not establish performance on newer articles, unfamiliar outlets, or changing topics. Evaluate on a time-based and source-separated test set, then add an external dataset before relying on the score.
- The dataset and training script should be audited for near-duplicates and conflicting labels across train and test sets. Split by source or story before fitting TF-IDF to reduce leakage risk.
- Linear SVM decision margins are not probabilities. The app shows a margin for that model; if a calibrated probability is needed, calibrate it on a separate validation set and test calibration quality.
- The term explanation shows which learned text features pushed the linear model toward a label. These features are not evidence for the underlying claims.
- Prediction history is temporary and session-scoped. Add persistent storage only if users need cross-session records, with a clear retention policy.
- Only scikit-learn is pinned because the model pickle records version 1.9.1; the remaining packages are not pinned. Record the full Python and package versions alongside each model export for repeatable deployments.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Include `app.py`, `requirements.txt`, `.streamlit/config.toml`, and the three model artifacts above. Include `train_model.py` for project reproducibility. The training CSV files are not needed to run the app and remain excluded by `.gitignore`.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, and set the app file to `app.py`.
4. Select **Deploy**. Streamlit Cloud installs `requirements.txt` and starts the app.

The app uses the NLTK English stopword corpus when it is installed. If the corpus is missing, it uses a small built-in fallback so the UI can still start; results may differ slightly from the training preprocessing until the corpus is available. Prediction history is kept in the current browser session and is not a permanent database.
