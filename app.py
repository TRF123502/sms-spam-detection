import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# download required nltk data
nltk.download("punkt")
nltk.download("stopwords")

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ================= TEXT TRANSFORMATION FUNCTION =================
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    cleaned_words = []
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            cleaned_words.append(ps.stem(word))

    return " ".join(cleaned_words)

# ================= LOAD MODEL & VECTORIZER =================
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# ================= STREAMLIT UI =================
st.title("📩 SMS Spam Detection App")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("🚨 This is a SPAM message")
    else:
        st.success("✅ This is NOT a spam message")
