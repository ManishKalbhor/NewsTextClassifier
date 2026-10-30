from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    from nltk.corpus import stopwords

    STOP_WORDS = set(stopwords.words("english"))
except (ImportError, LookupError):
    # Keeps the app usable on a fresh/offline deployment. Install the NLTK
    # stopwords corpus to match the training pre-processing exactly.
    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
        "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
        "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
        "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
        "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
        "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
        "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your",
        "yours", "yourself", "yourselves",
    }


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "fake_news_model.pkl"
VECTORIZER_PATH = APP_DIR / "model" / "tfidf_vectorizer.pkl"
METRICS_PATH = APP_DIR / "model" / "metrics.json"

st.set_page_config(
    page_title="FakeNewsTextClassifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #EAF2F0;
        --teal: #42C8B2;
        --teal-dark: #168E82;
        --coral: #FF9A76;
        --line: #2C414A;
        --paper: #101A21;
    }

    .stApp {
        color: var(--ink);
        background:
            radial-gradient(ellipse at 95% 0%, rgba(66, 200, 178, 0.13), transparent 38rem),
            var(--paper);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14232B 0%, #101A21 100%);
        border-right: 1px solid var(--line);
    }

    h1 {
        color: #F4FAF8;
        letter-spacing: -0.045em;
    }

    h2, h3 {
        color: #BCECE3;
        letter-spacing: -0.025em;
    }

    hr {
        height: 1px;
        border: 0;
        background: linear-gradient(90deg, transparent, var(--teal), var(--coral), transparent);
    }

    .stButton > button {
        border-radius: 12px;
        transition: transform 160ms ease, box-shadow 160ms ease, background-color 160ms ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 18px rgba(20, 55, 58, 0.14);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--teal-dark), #126F69);
        border-color: var(--teal);
        color: #FFFFFF;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1A2B33, #15242C);
        border: 1px solid var(--line);
        border-radius: 16px;
        box-shadow: 0 3px 14px rgba(0, 0, 0, 0.16);
        transition: transform 180ms ease, box-shadow 180ms ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28);
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--teal);
        box-shadow: 0 0 0 3px rgba(66, 200, 178, 0.20);
    }

    [data-testid="stRadio"] label {
        border-radius: 10px;
        transition: background-color 150ms ease, color 150ms ease;
    }

    [data-testid="stRadio"] label:hover {
        color: var(--teal-dark);
        background: rgba(66, 200, 178, 0.10);
    }

    main .block-container {
        animation: page-arrive 380ms ease-out both;
    }

    @keyframes page-arrive {
        from { opacity: 0; transform: translateY(7px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(word for word in text.split() if word not in STOP_WORDS)


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def explain_prediction(model, vector, vectorizer):
    if not hasattr(model, "coef_"):
        return None
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    row = vector.tocoo()
    items = [
        {"Term": feature_names[index], "Influence": float(value * coefficients[index])}
        for index, value in zip(row.col, row.data)
    ]
    items.sort(key=lambda item: abs(item["Influence"]), reverse=True)
    return items[:8]


missing = [path for path in (MODEL_PATH, VECTORIZER_PATH, METRICS_PATH) if not path.exists()]
if missing:
    st.error("The app is missing required model files: " + ", ".join(path.name for path in missing))
    st.info("Run `python train_model.py` from the project folder to create the model artifacts.")
    st.stop()

try:
    model, vectorizer = load_artifacts()
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
except Exception as exc:
    st.error(f"Could not load the model artifacts: {exc}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.markdown("### 📰 FakeNewsTextClassifier")
    st.caption("NEWS TEXT ANALYSIS")
    st.divider()
    page_choice = st.radio(
        "Workspace",
        ["🧪  Detector", "📊  Overview", "📈  Performance", "🕘  History"],
        label_visibility="collapsed",
    )
    page = {
        "🧪  Detector": "Detector",
        "📊  Overview": "Overview",
        "📈  Performance": "Model performance",
        "🕘  History": "History",
    }[page_choice]
    st.divider()
    st.caption("Powered by TF-IDF and scikit-learn")

st.title("FakeNewsTextClassifier")
st.caption("A text based screening tool for headlines and articles. Use its output as a starting point for research, not a fact check.")

if page == "Detector":
    st.divider()
    st.subheader("Analyze a news story")
    st.write("Enter a headline, article, or both. Adding the full article gives the model more text to analyze.")
    with st.form("analysis_form", clear_on_submit=False):
        headline = st.text_input("Headline", placeholder="Paste the story headline", max_chars=500)
        article = st.text_area("Article text", height=220, placeholder="Paste the article text here", max_chars=50000)
        submitted = st.form_submit_button("Analyze story", type="primary", use_container_width=True)

    if submitted:
        st.session_state.last_result = None
        if not headline.strip() and not article.strip():
            st.warning("Enter a headline or article before analyzing.")
        else:
            cleaned = clean_text(f"{headline} {article}")
            if not cleaned:
                st.warning("There is not enough usable text to analyze. Try adding more of the story.")
            else:
                vector = vectorizer.transform([cleaned])
                if vector.nnz == 0:
                    st.warning("The model did not recognize any words from its training vocabulary. Try adding more text or rephrasing it.")
                else:
                    prediction = int(model.predict(vector)[0])
                    label = "REAL" if prediction == 1 else "FAKE"
                    # LinearSVC decision scores are not calibrated probabilities.
                    # Keep the output honest by describing score strength, not probability.
                    decision_score = None
                    if hasattr(model, "decision_function"):
                        decision_score = float(model.decision_function(vector)[0])
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(vector)[0]
                        class_order = list(model.classes_)
                        confidence = float(probabilities[class_order.index(prediction)])
                        score_label = "Model score"
                        score_value = confidence * 100
                        score_note = "The model's score is not a verified likelihood that the story is true or false."
                    else:
                        score_label = "Decision margin"
                        score_value = abs(decision_score) if decision_score is not None else None
                        score_note = "Margin shows distance from the model's boundary; it is not a probability."
                    result = {
                        "label": label,
                        "headline": headline.strip() or "(No headline provided)",
                        "score_label": score_label,
                        "score_value": score_value,
                        "score_note": score_note,
                        "explanations": explain_prediction(model, vector, vectorizer),
                        "time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
                    }
                    st.session_state.last_result = result
                    st.session_state.history.insert(0, {
                        "Time": result["time"],
                        "Headline": result["headline"][:120],
                        "Prediction": label,
                        "Score type": score_label,
                        "Score": (f"{score_value:.1f}%" if score_label == "Model score" else f"{score_value:.3f}") if score_value is not None else "—",
                    })
                    st.session_state.history = st.session_state.history[:100]

    result = st.session_state.last_result
    if result:
        if result["label"] == "FAKE":
            st.error("The model flags this text as FAKE")
        else:
            st.success("The model flags this text as REAL")
        st.caption("This label is based on text patterns learned from the training data. Verify the source and claims independently.")
        left, right = st.columns([1, 2])
        with left:
            if result["score_value"] is not None:
                st.metric(result["score_label"], f"{result['score_value']:.2f}" if result["score_label"] == "Decision margin" else f"{result['score_value']:.1f}%")
            st.caption(result["score_note"])
        with right:
            st.markdown("**What influenced the label**")
            if result["explanations"]:
                explain_df = pd.DataFrame(result["explanations"])
                explain_df["Direction"] = explain_df["Influence"].apply(lambda value: "Real" if value > 0 else "Fake")
                explain_df["Strength"] = explain_df["Influence"].abs().round(4)
                st.dataframe(explain_df[["Term", "Direction", "Strength"]], use_container_width=True, hide_index=True)
                st.caption("These terms show model influence, not evidence that a claim is true or false.")
            else:
                st.info("Term-level explanations are unavailable for this model.")

elif page == "Overview":
    st.divider()
    dataset = metrics["dataset"]
    best_model = metrics["best_model"]
    best_accuracy = metrics["results"][best_model]["accuracy"] * 100
    st.subheader("Model at a glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test accuracy", f"{best_accuracy:.2f}%")
    col2.metric("Selected model", best_model)
    col3.metric("Training records", f"{dataset['training_articles']:,}")
    col4.metric("Held-out records", f"{dataset['testing_articles']:,}")
    st.subheader("Raw source dataset")
    chart_data = pd.DataFrame({"Label": ["Fake", "Real"], "Articles": [dataset["fake_articles"], dataset["real_articles"]]}).set_index("Label")
    st.bar_chart(chart_data)
    st.caption(f"Source files contain {dataset['fake_articles']:,} fake-labeled and {dataset['real_articles']:,} real-labeled rows. After cleaning and deduplication, {dataset['total_articles']:,} records were used.")
    st.info("Accuracy can be inflated by duplicated or source-specific patterns. Test on newer, independent sources before relying on this model.")

elif page == "Model performance":
    st.divider()
    st.subheader("Compare model results")
    comparison = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": round(result["accuracy"] * 100, 2),
            "Macro F1": round(result["classification_report"]["macro avg"]["f1-score"] * 100, 2),
        }
        for name, result in metrics["results"].items()
    ]).sort_values("Accuracy", ascending=False)
    left, right = st.columns([1.2, 1])
    with left:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.bar_chart(comparison.set_index("Model")[["Accuracy", "Macro F1"]])
    with right:
        selected_model = st.selectbox("Inspect model", list(metrics["results"].keys()), index=list(metrics["results"].keys()).index(metrics["best_model"]))
        result = metrics["results"][selected_model]
        cm = result["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#17262E")
        ax.set_facecolor("#17262E")
        ax.imshow(cm, cmap="YlGnBu")
        ax.set_title("Confusion matrix", pad=14, color="#EAF2F0")
        ax.set_xlabel("Predicted label", color="#C4D5D2")
        ax.set_ylabel("Actual label", color="#C4D5D2")
        ax.tick_params(colors="#C4D5D2")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Fake", "Real"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Fake", "Real"])
        for row in range(2):
            for col in range(2):
                value = cm[row][col]
                text_color = "#102027" if value < max(max(values) for values in cm) * 0.55 else "#FFFFFF"
                ax.text(col, row, f"{value:,}", ha="center", va="center", color=text_color, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    report = pd.DataFrame(metrics["results"][selected_model]["classification_report"]).transpose()
    st.subheader("Class-level results")
    st.dataframe(report.round(3), use_container_width=True)
    st.caption("The evaluation uses the project's random train/test split. It does not measure robustness to time, source, or topic shifts.")

else:
    st.divider()
    st.subheader("Recent analyses")
    if not st.session_state.history:
        st.info("Your analyses will appear here during this session.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        csv = history_df.to_csv(index=False).encode("utf-8")
        left, right = st.columns([1, 5])
        with left:
            st.download_button("Download CSV", csv, "signal-analysis-history.csv", "text/csv", use_container_width=True)
        with right:
            if st.button("Clear history", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        st.caption("History is stored temporarily in this browser session and is not sent to a database.")

st.divider()
st.caption("FakeNewsTextClassifier · Always verify claims with reliable, independent sources")
