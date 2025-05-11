import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords if needed
nltk.download('stopwords')
arabic_stopwords = set(stopwords.words('arabic'))

# Clean text manually here
def normalize_arabic(text):
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\u0600-ۿ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_arabic_text(text):
    text = normalize_arabic(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in arabic_stopwords]
    return ' '.join(tokens)

# Load model and vectorizer
with open("MLP_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit Interface
st.set_page_config(page_title="NLP - Problem Source Detection", layout="centered")
st.title("🧠 NLP - Detection of the Problem Source")
st.write("This tool uses NLP to detect whether a problem is related to **الخدمة (service)** or **المنتج (product)** based on Arabic customer feedback.")

# Input
text_input = st.text_area("✍️ Enter Arabic customer feedback:", height=150)

if st.button("🔍 Detect Problem Source"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_arabic_text(text_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        label = prediction[0]
        st.success(f"🧾 The predicted problem source is: **{label}**")
