import streamlit as st
import joblib
import re
import nltk
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from nltk.corpus import stopwords


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fake News AI Detector",
    page_icon="📰",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.main-title {
    font-size: 45px;
    font-weight: 800;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #9aa4b2;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 25px;
    margin-bottom: 20px;
}

.fake {
    background-color: #3b171c;
    border: 1px solid #8e2b35;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
}

.real {
    background-color: #123524;
    border: 1px solid #2d8a57;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
}

.result-title {
    font-size: 30px;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# NLTK
# =========================================================

nltk.download(
    "stopwords",
    quiet=True
)

STOP_WORDS = set(
    stopwords.words("english")
)


# =========================================================
# FILE PATHS
# =========================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "fake_news_model.pkl"
VECTORIZER_PATH = APP_DIR / "model" / "tfidf_vectorizer.pkl"
METRICS_PATH = APP_DIR / "model" / "metrics.json"


# =========================================================
# CHECK FILES
# =========================================================

if not os.path.exists(
    MODEL_PATH
):

    st.error(
        "Model not found. "
        "Run python train_model.py"
    )

    st.stop()


if not os.path.exists(
    VECTORIZER_PATH
):

    st.error(
        "TF-IDF vectorizer not found."
    )

    st.stop()


if not os.path.exists(
    METRICS_PATH
):

    st.error(
        "metrics.json not found."
    )

    st.stop()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


model, vectorizer = load_artifacts()


# =========================================================
# LOAD METRICS
# =========================================================

with open(
    METRICS_PATH,
    "r"
) as file:

    metrics = json.load(file)


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
# PROBABILITY
# =========================================================

def get_probability(
    model,
    vector
):

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                vector
            )[0]
        )

        return (
            probabilities[0],
            probabilities[1]
        )

    else:

        score = (
            model.decision_function(
                vector
            )[0]
        )

        real_probability = (
            1 /
            (
                1 +
                pow(
                    2.71828,
                    -score
                )
            )
        )

        fake_probability = (
            1 -
            real_probability
        )

        return (
            fake_probability,
            real_probability
        )


# =========================================================
# EXPLAIN PREDICTION
# =========================================================

def explain_prediction(
    model,
    vector
):

    if not hasattr(
        model,
        "coef_"
    ):

        return None

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    coefficients = (
        model.coef_[0]
    )

    row = vector.tocoo()

    explanations = []

    for index, value in zip(
        row.col,
        row.data
    ):

        contribution = (
            value *
            coefficients[index]
        )

        explanations.append({

            "Word / Phrase":
                feature_names[index],

            "Influence":
                contribution

        })

    explanations.sort(
        key=lambda x:
        abs(x["Influence"]),
        reverse=True
    )

    return explanations[:10]


# =========================================================
# SESSION HISTORY
# =========================================================

if "history" not in st.session_state:

    st.session_state.history = []


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '📰 Fake News AI Detector'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Fake News Classification using NLP'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title(
        "📰 Fake News AI"
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Detector",
            "📊 Dashboard",
            "📈 Model Performance",
            "🕒 Prediction History"
        ]
    )

    st.markdown("---")

    st.write(
        "### 🤖 Technologies"
    )

    st.write(
        """
        Python  
        NLP  
        TF-IDF  
        Machine Learning  
        Streamlit
        """
    )


# =========================================================
# DETECTOR
# =========================================================

if page == "🏠 Detector":

    st.subheader(
        "📰 Analyze News"
    )

    # -----------------------------------------------------
    # SEPARATE HEADLINE
    # -----------------------------------------------------

    headline = st.text_input(
        "📝 News Headline",
        placeholder=
        "Enter news headline..."
    )


    # -----------------------------------------------------
    # SEPARATE ARTICLE
    # -----------------------------------------------------

    article = st.text_area(
        "📄 News Article",
        height=280,
        placeholder=
        "Paste the complete news article here..."
    )


    # -----------------------------------------------------
    # BUTTON
    # -----------------------------------------------------

    analyze = st.button(
        "🔍 Analyze News",
        type="primary",
        use_container_width=True
    )


    if analyze:

        if not headline and not article:

            st.warning(
                "Please enter headline or article."
            )

        else:

            combined_text = (
                headline +
                " " +
                article
            )

            cleaned = clean_text(
                combined_text
            )

            vector = vectorizer.transform(
                [cleaned]
            )


            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------

            prediction = model.predict(
                vector
            )[0]


            fake_prob, real_prob = (
                get_probability(
                    model,
                    vector
                )
            )


            fake_percent = (
                fake_prob * 100
            )

            real_percent = (
                real_prob * 100
            )


            confidence = max(
                fake_percent,
                real_percent
            )


            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            if prediction == 0:

                st.markdown(
                    """
                    <div class="fake">

                    <div class="result-title">
                    ❌ FAKE NEWS
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                label = "FAKE"

            else:

                st.markdown(
                    """
                    <div class="real">

                    <div class="result-title">
                    ✅ REAL NEWS
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                label = "REAL"


            # -------------------------------------------------
            # PROBABILITIES
            # -------------------------------------------------

            st.subheader(
                "📊 Prediction Probability"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "❌ Fake",
                    f"{fake_percent:.2f}%"
                )

            with col2:

                st.metric(
                    "✅ Real",
                    f"{real_percent:.2f}%"
                )

            with col3:

                st.metric(
                    "🎯 Confidence",
                    f"{confidence:.2f}%"
                )


            st.progress(
                int(confidence)
            )


            # -------------------------------------------------
            # WHY?
            # -------------------------------------------------

            st.markdown("---")

            st.subheader(
                "🧠 Why did the model classify it this way?"
            )


            explanations = (
                explain_prediction(
                    model,
                    vector
                )
            )


            if explanations:

                explanation_df = (
                    pd.DataFrame(
                        explanations
                    )
                )

                explanation_df[
                    "Direction"
                ] = explanation_df[
                    "Influence"
                ].apply(
                    lambda x:
                    "REAL" if x > 0
                    else "FAKE"
                )

                explanation_df[
                    "Influence"
                ] = explanation_df[
                    "Influence"
                ].abs().round(4)

                explanation_df = (
                    explanation_df[
                        [
                            "Word / Phrase",
                            "Direction",
                            "Influence"
                        ]
                    ]
                )

                st.dataframe(
                    explanation_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "These words/features are influential "
                    "according to the trained model. "
                    "They are not factual proof."
                )

            else:

                st.info(
                    "Explanation is not available "
                    "for this model."
                )


            # -------------------------------------------------
            # HISTORY
            # -------------------------------------------------

            st.session_state.history.insert(
                0,
                {
                    "Time":
                        pd.Timestamp.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "Headline":
                        headline[:100],

                    "Prediction":
                        label,

                    "Confidence":
                        f"{confidence:.2f}%"
                }
            )


# =========================================================
# DASHBOARD
# =========================================================

elif page == "📊 Dashboard":

    st.title(
        "📊 Model Dashboard"
    )


    best_model = metrics[
        "best_model"
    ]

    best_accuracy = (
        metrics[
            "results"
        ][
            best_model
        ][
            "accuracy"
        ] * 100
    )


    dataset = metrics[
        "dataset"
    ]


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(
            "🎯 Accuracy",
            f"{best_accuracy:.2f}%"
        )


    with col2:

        st.metric(
            "🏆 Best Model",
            best_model
        )


    with col3:

        st.metric(
            "❌ Fake Articles",
            dataset[
                "fake_articles"
            ]
        )


    with col4:

        st.metric(
            "✅ Real Articles",
            dataset[
                "real_articles"
            ]
        )


    st.markdown("---")

    st.subheader(
        "📚 Dataset"
    )

    st.write(
        "Total Articles:",
        dataset[
            "total_articles"
        ]
    )

    st.write(
        "Training Articles:",
        dataset[
            "training_articles"
        ]
    )

    st.write(
        "Testing Articles:",
        dataset[
            "testing_articles"
        ]
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📈 Model Performance":

    st.title(
        "📈 Model Performance"
    )


    # -----------------------------------------------------
    # COMPARISON TABLE
    # -----------------------------------------------------

    comparison = []

    for name, result in (
        metrics[
            "results"
        ].items()
    ):

        comparison.append({

            "Model":
                name,

            "Accuracy":
                round(
                    result[
                        "accuracy"
                    ] * 100,
                    2
                )
        })


    comparison_df = pd.DataFrame(
        comparison
    )


    st.subheader(
        "🔄 Model Accuracy Comparison"
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # BAR CHART
    # -----------------------------------------------------

    st.bar_chart(
        comparison_df.set_index(
            "Model"
        )
    )


    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📊 Confusion Matrix"
    )


    selected_model = st.selectbox(
        "Select Model",
        list(
            metrics[
                "results"
            ].keys()
        )
    )


    cm = metrics[
        "results"
    ][
        selected_model
    ][
        "confusion_matrix"
    ]


    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_title(
        selected_model
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )


    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        ["Fake", "Real"]
    )

    ax.set_yticklabels(
        ["Fake", "Real"]
    )


    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                cm[i][j],
                ha="center",
                va="center"
            )


    st.pyplot(
        fig
    )


    # -----------------------------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------------------------

    st.subheader(
        "📋 Classification Report"
    )


    report = metrics[
        "results"
    ][
        selected_model
    ][
        "classification_report"
    ]


    report_df = pd.DataFrame(
        report
    ).transpose()


    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )


# =========================================================
# HISTORY
# =========================================================

elif page == "🕒 Prediction History":

    st.title(
        "🕒 Prediction History"
    )


    if not st.session_state.history:

        st.info(
            "No predictions made yet."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )


        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )


        # -------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------

        csv = history_df.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            "⬇️ Download History",
            csv,
            "prediction_history.csv",
            "text/csv",
            use_container_width=True
        )


        # -------------------------------------------------
        # CLEAR
        # -------------------------------------------------

        if st.button(
            "🗑️ Clear History"
        ):

            st.session_state.history = []

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "📰 Fake News AI Detector | "
    "Python • NLP • TF-IDF • Machine Learning • Streamlit"
)
