import streamlit as st
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Download NLTK stopwords if not already present
nltk.download('stopwords')

# ========== TRAINING PART (reuse Day 4 logic, run once, then comment out for faster loading) ==========
# If you already have 'model.joblib' and 'vectorizer.joblib', you can skip retraining for faster loading.

CSV_PATH = 'IMDB_Dataset.csv'  # Adjust if your dataset filename is different
TEXT_COL = 'review'
LABEL_COL = 'sentiment'

def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

if not (os.path.exists("model.joblib") and os.path.exists("vectorizer.joblib")):
    df = pd.read_csv(CSV_PATH)
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
    df[LABEL_COL] = df[LABEL_COL].map({'positive': 1, 'negative': 0})
    X = df[TEXT_COL]
    y = df[LABEL_COL]
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)
    joblib.dump(model, "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")

# ========== LOAD TRAINED MODEL AND VECTORIZER ==========
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# ========== STREAMLIT APP ==========
st.set_page_config(page_title="Movie Review Sentiment Analyzer")
st.title("🎬 Movie Review Sentiment Analyzer")

review = st.text_area("Enter your movie review:", height=120)
if st.button("Predict"):
    cleaned = clean_text(review)
    vec = vectorizer.transform([cleaned])
    pred_proba = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]
    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    confidence = pred_proba[pred]
    # Emoji + Color
    if pred == 1:
        st.success(f"😊 Result: **{sentiment}** (Confidence: {confidence*100:.2f}%)")
        st.markdown('<style>div.stAlert{background-color:#d4edda;}</style>', unsafe_allow_html=True)
    else:
        st.error(f"😞 Result: **{sentiment}** (Confidence: {confidence*100:.2f}%)")
        st.markdown('<style>div.stAlert{background-color:#f8d7da;}</style>', unsafe_allow_html=True)

# Optional: Add some quick test reviews
with st.expander("Try example reviews"):
    st.write('''
- "This was the worst movie ever."
- "I enjoyed every minute of it!"
- "The plot was predictable and boring."
- "Amazing performance and storyline!"
''')
