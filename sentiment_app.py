import streamlit as st
import numpy as np
import re
import pickle

st.set_page_config(
    page_title="Tweet Sentiment",
    page_icon="🐦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;1,400&family=DM+Serif+Display&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #f8f9fa !important;
    color: #212529 !important;
}

.stApp {
    background-color: #f8f9fa !important;
}

/* force all streamlit containers white/light */
section[data-testid="stSidebar"],
.css-1d391kg, .css-12oz5g7,
div[data-testid="stAppViewContainer"],
div[data-testid="stHeader"] {
    background-color: #f8f9fa !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 660px; }

/* override streamlit dark inputs */
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #212529 !important;
    border: 1px solid #dee2e6 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}

.stButton > button {
    background-color: #0466c8 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1rem !important;
}

.stButton > button:hover {
    background-color: #0353a4 !important;
}

/* ── header ── */
.site-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.1rem;
    color: #001233;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.site-sub {
    font-size: 0.88rem;
    color: #6c757d;
    margin: 0 0 14px 0;
}
.blue-rule {
    width: 100%;
    height: 2px;
    background: #0466c8;
    border: none;
    margin: 0 0 16px 0;
}

/* ── chips ── */
.chip-row { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:18px; }
.chip {
    background: #e9ecef;
    border: 1px solid #ced4da;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.74rem;
    color: #495057;
    font-weight: 500;
}

/* ── section label ── */
.section-label {
    font-size: 0.71rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #adb5bd;
    margin: 0 0 8px 0;
}

/* ── result card ── */
.card {
    border-radius: 10px;
    padding: 18px 20px;
    margin: 16px 0 10px 0;
    border-left: 4px solid;
}
.card-positive   { background: #e8f4fd; border-color: #0466c8; }
.card-negative   { background: #fff0f0; border-color: #c1121f; }
.card-neutral    { background: #f8f9fa; border-color: #6c757d; border: 1px solid #dee2e6; border-left: 4px solid #6c757d; }
.card-irrelevant { background: #f8f9fa; border-color: #adb5bd; border: 1px solid #dee2e6; border-left: 4px solid #adb5bd; }

.card-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #001233;
    margin: 0 0 3px 0;
}
.card-conf { font-size: 0.82rem; color: #6c757d; margin: 0; }

/* ── bars ── */
.bars-wrap { margin: 14px 0 0 0; }
.bar-row { display:flex; align-items:center; gap:8px; margin:5px 0; }
.bar-name { width:82px; font-size:0.78rem; color:#6c757d; text-align:right; flex-shrink:0; }
.bar-name-active { width:82px; font-size:0.78rem; color:#001233; font-weight:600; text-align:right; flex-shrink:0; }
.bar-track { flex:1; background:#e9ecef; border-radius:4px; height:6px; overflow:hidden; }
.bar-fill  { height:100%; border-radius:4px; }
.bar-pct   { width:38px; font-size:0.77rem; color:#495057; flex-shrink:0; }

/* ── cleaned box ── */
.clean-wrap {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 10px;
}
.clean-meta { font-size:0.7rem; color:#adb5bd; margin-bottom:4px; }
.clean-text { font-size:0.82rem; color:#495057; font-family:monospace; }

/* ── footer ── */
.footer {
    border-top: 1px solid #dee2e6;
    margin-top: 28px;
    padding-top: 10px;
    font-size: 0.72rem;
    color: #adb5bd;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
CLASSES = ['Irrelevant', 'Negative', 'Neutral', 'Positive']
MAX_LEN = 50

CARD_CSS = {
    'Positive':   ('card-positive',   '😊', '#0466c8'),
    'Negative':   ('card-negative',   '😠', '#c1121f'),
    'Neutral':    ('card-neutral',    '😐', '#495057'),
    'Irrelevant': ('card-irrelevant', '💬', '#6c757d'),
}
BAR_COLOR = {
    'Positive':   '#0466c8',
    'Negative':   '#c1121f',
    'Neutral':    '#6c757d',
    'Irrelevant': '#adb5bd',
}

# ── load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load():
    try:
        import tensorflow as tf
        m = tf.keras.models.load_model('best_sentiment_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tok = pickle.load(f)
        return m, tok, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def get_nlp():
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    lem   = WordNetLemmatizer()
    stops = set(stopwords.words('english'))
    neg   = {"not","no","never","nor","neither","without",
             "don't","doesn't","didn't","won't","can't",
             "isn't","wasn't","weren't","shouldn't","wouldn't"}
    stops -= neg
    return lem, stops

def clean(text, lem, stops):
    if not isinstance(text, str) or not text.strip(): return ''
    t = text.lower()
    t = re.sub(r'http\S+|www\S+', '', t)
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'#(\w+)', r'\1', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return ' '.join([lem.lemmatize(w) for w in t.split() if w not in stops and len(w)>1])

def predict(text, model, tok):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    lem, stops = get_nlp()
    cleaned    = clean(text, lem, stops)
    seq        = tok.texts_to_sequences([cleaned])
    padded     = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    probs      = model.predict(padded, verbose=0)[0]
    idx        = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), dict(zip(CLASSES, probs.tolist())), cleaned

# ── header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="site-title">Tweet Sentiment</p>', unsafe_allow_html=True)
st.markdown('<p class="site-sub">Paste any tweet — the model tells you if it\'s positive, negative, neutral, or irrelevant.</p>', unsafe_allow_html=True)
st.markdown('<hr class="blue-rule">', unsafe_allow_html=True)
st.markdown("""
<div class="chip-row">
  <span class="chip">Bi-LSTM</span>
  <span class="chip">86.2% accuracy</span>
  <span class="chip">0.97 AUC</span>
  <span class="chip">4 classes</span>
  <span class="chip">TensorFlow · Keras</span>
</div>
""", unsafe_allow_html=True)

# ── load ──────────────────────────────────────────────────────────────────────
model, tokenizer, err = load()
if err:
    st.error(f"Could not load model: {err}")
    st.info("Make sure `best_sentiment_model.h5` and `tokenizer.pkl` are in the same folder.")
    st.stop()

# ── examples ──────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Quick examples</p>', unsafe_allow_html=True)
examples = {
    "Positive 😊": "I absolutely love this! Best thing ever.",
    "Negative 😠": "Completely broken and useless. Never again.",
    "Neutral 😐":  "The new update was released today.",
    "Irrelevant 💬": "I had rice for lunch today.",
}
cols = st.columns(4)
for col, (label, tweet) in zip(cols, examples.items()):
    if col.button(label, use_container_width=True):
        st.session_state['tw'] = tweet

# ── input ─────────────────────────────────────────────────────────────────────
tweet_input = st.text_area(
    "Tweet",
    value=st.session_state.get('tw', ''),
    placeholder="Type or paste a tweet here...",
    height=100,
    label_visibility="collapsed"
)
run = st.button("Analyze sentiment →", type="primary", use_container_width=True)

# ── result ────────────────────────────────────────────────────────────────────
if run:
    if not tweet_input.strip():
        st.warning("Please enter something first.")
    else:
        with st.spinner(""):
            label, conf, all_probs, cleaned = predict(tweet_input, model, tokenizer)

        css, emoji, color = CARD_CSS[label]

        st.markdown(f"""
        <div class="card {css}">
            <p class="card-label">{emoji} {label}</p>
            <p class="card-conf">{conf:.1%} confidence</p>
        </div>
        """, unsafe_allow_html=True)

        bars = '<div class="bars-wrap">'
        for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            w        = prob * 100
            name_cls = 'bar-name-active' if cls == label else 'bar-name'
            bars    += f"""
            <div class="bar-row">
                <div class="{name_cls}">{cls}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{w:.1f}%;background:{BAR_COLOR[cls]}"></div>
                </div>
                <div class="bar-pct">{prob:.1%}</div>
            </div>"""
        bars += '</div>'
        st.markdown(bars, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="clean-wrap">
            <div class="clean-meta">preprocessed input</div>
            <div class="clean-text">{cleaned if cleaned else "(empty after cleaning)"}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    BSE-634 · Bi-LSTM · Twitter Entity Sentiment Dataset · 86.2% test accuracy
</div>
""", unsafe_allow_html=True)
