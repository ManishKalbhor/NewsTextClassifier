# Fake News AI Detector

A Streamlit app that classifies a news headline and article with a trained TF-IDF model.

## Run locally

Use Python 3.10 or 3.11, then run these commands from the project folder:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app needs these committed artifacts in `model/`:

- `fake_news_model.pkl`
- `tfidf_vectorizer.pkl`
- `metrics.json`

If you retrain the model, run `python train_model.py` and commit the updated artifacts. Pickle files must be loaded with compatible versions of scikit-learn and its dependencies; if deployment reports a model version error, retrain and deploy the artifacts using the dependency versions in `requirements.txt`.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Include `app.py`, `requirements.txt`, and the three model artifacts above. The training CSV files are not needed to run the app.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, and set the app file to `app.py`.
4. Select **Deploy**. Streamlit Cloud installs `requirements.txt` and starts the app.

The app downloads the NLTK English stopword corpus when it starts, so its first startup needs network access. The prediction history is kept in the current browser session and is not a permanent database.
