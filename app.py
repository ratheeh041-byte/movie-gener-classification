import string

import joblib
import pandas as pd
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(page_title="Movie Genre Predictor", page_icon="🎬", layout="centered")

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #ff9d4b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .result-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #ff4b4b22, #ff9d4b22);
        border: 1px solid #ff4b4b55;
        margin: 1rem 0;
    }
    .result-genre {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    .result-conf {
        color: #888;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load model ----------
@st.cache_resource
def load_files():
    return joblib.load("model.pkl"), joblib.load("tfidf.pkl")


try:
    model, tfidf = load_files()
except FileNotFoundError:
    st.error("model.pkl or tfidf.pkl not found. Run `python train.py` first.")
    st.stop()

EMOJI = {
    "drama": "🎭", "thriller": "😱", "adult": "🔞", "documentary": "🎥",
    "comedy": "😂", "crime": "🕵️", "reality-tv": "📺", "horror": "👻",
    "sport": "⚽", "animation": "🎨", "action": "💥", "fantasy": "🧙",
    "short": "⏱️", "sci-fi": "🚀", "music": "🎵", "adventure": "🗺️",
    "talk-show": "🎤", "western": "🤠", "family": "👨‍👩‍👧", "mystery": "🔍",
    "history": "📜", "news": "📰", "biography": "📖", "romance": "💕",
    "game-show": "🎮", "musical": "🎶", "war": "🪖",
}

EXAMPLES = {
    "🔪 Crime": "A detective hunts a serial killer in a dark city while the police search for clues.",
    "💕 Romance": "Two strangers fall in love before her wedding and must choose between love and family.",
    "🚀 Sci-Fi": "A crew of astronauts travels to a distant planet and discovers an alien spaceship.",
    "😂 Comedy": "Three friends go on a hilarious road trip and get into funny trouble everywhere they go.",
    "🎭 Drama": "A family struggles with loss and betrayal as they try to rebuild their lives after a tragedy.",
    "🧙 Fantasy": "A young wizard attends a magical school and must stop an evil sorcerer from taking over the world.",
    "👻 Horror": "A group of friends stay in a haunted house and are terrorized by ghosts and demons.",
    "🎨 Animation": "A young boy befriends a talking animal and goes on a magical adventure to save his village.",
    "🎵 Musical": "A struggling singer gets a big break and must overcome obstacles to achieve her dreams.",
    "🕵️ Thriller": "A spy must stop a terrorist plot while being hunted by his own agency.",
    "📰 News": "A journalist investigates a political scandal and uncovers corruption at the highest levels.",
    "📜 History": "A historical drama about the life of a famous leader and the events that shaped their legacy.",
}


def clean(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def set_example(text):
    st.session_state.plot = text


# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app reads a movie's story and predicts its genre "
        "using **TF-IDF** and **Logistic Regression**."
    )
    st.metric("Genres known", len(model.classes_))
    st.caption("Tip: longer plots (30+ words) give better predictions.")
    top_n = st.slider("Genres to show", 3, 8, 5)

# ---------- Main page ----------
st.markdown('<p class="title">🎬 Movie Genre Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Paste a movie story and let AI guess the genre</p>',
    unsafe_allow_html=True,
)

if "plot" not in st.session_state:
    st.session_state.plot = ""

st.write("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES.items()):
    col.button(label, on_click=set_example, args=(text,), use_container_width=True)

plot = st.text_area(
    "Enter the movie plot:",
    key="plot",
    height=160,
    placeholder="Example: A young boy discovers he has magical powers...",
)

if st.button("🔮 Predict Genre", type="primary", use_container_width=True):
    if plot.strip() == "":
        st.warning("Please enter a plot first.")
    else:
        if len(plot.split()) < 10:
            st.info("Short plots are harder to predict. Try adding more detail.")

        with st.spinner("Analysing the story..."):
            X = tfidf.transform([clean(plot)])
            probs = model.predict_proba(X)[0]

        result = (
            pd.DataFrame({"Genre": model.classes_, "Confidence": probs})
            .sort_values("Confidence", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        best = result.iloc[0]
        emoji = EMOJI.get(best["Genre"], "🎞️")
        st.markdown(
            f"""
            <div class="result-card">
                <p class="result-conf">Predicted genre</p>
                <p class="result-genre">{emoji} {best['Genre'].title()}</p>
                <p class="result-conf">Confidence: {best['Confidence'] * 100:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Top predictions")
        for _, row in result.iterrows():
            e = EMOJI.get(row["Genre"], "🎞️")
            st.write(f"{e} **{row['Genre'].title()}**  ({row['Confidence'] * 100:.1f}%)")
            st.progress(min(int(row["Confidence"] * 100), 100))

        st.bar_chart(result.set_index("Genre")["Confidence"])

st.divider()
st.caption("College project | Movie Genre Classification using Machine Learning")
