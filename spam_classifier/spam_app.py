# spam_app.py
# Paste into spam_app.py in the project root (same folder as spam_pipeline.joblib)

import streamlit as st
import joblib, re, string

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@st.cache_resource
def load_model():
    return joblib.load("spam_pipeline.joblib")

pipeline = load_model()

st.title("📧📱 SMS & Email Spam Classifier")
st.write("Paste an SMS or an Email body and click **Check**.")

user_input = st.text_area("Enter message here:")

if st.button("Check"):
    if not user_input.strip():
        st.warning("Please paste a message.")
    else:
        cleaned = clean_text(user_input)
        pred = pipeline.predict([cleaned])[0]
        proba = pipeline.predict_proba([cleaned])[0][1]
        if pred == 1:
            st.error(f"🚨 SPAM (probability {proba:.2f})")
        else:
            st.success(f"✅ NOT SPAM (probability {proba:.2f})")
