
import streamlit as st
from analyzer import analyze_text
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["sentiment_db"]
collection = db["user_inputs"]

st.set_page_config(page_title="Emotional Signal Analyzer")
st.title("Emotional Signal Analyzer")

user_text = st.text_area(
    "Write what you're feeling:",
    height=160,
    placeholder="Type freely. There are no right words."
)

if st.button("Analyze"):
    if user_text.strip():
        result = analyze_text(user_text)

        # 🔥 STORE DATA IN MONGODB
        collection.insert_one({
            "text": user_text,
            "result": result,
            "timestamp": datetime.utcnow()
        })

        st.subheader("Detected Emotional Signal")
        st.json(result)
    else:
        st.warning("Please type something first.")

import streamlit as st
from pymongo import MongoClient
from urllib.parse import quote_plus
import datetime


username = "sentiment_db"
password = "nemesis007"
cluster = "cluster0.ci3lhyt"
database = "sentiment_db"

password_encoded = quote_plus(password)
uri = mongodb+srv://sentiment_db:nemesis007@cluster0.ci3lhyt.mongodb.net/sentiment_db

client = MongoClient(uri)
db = client[database]
collection = db["user_sentiments"]


def predict_sentiment(text):
    if "good" in text.lower():
        return "Positive"
    elif "bad" in text.lower():
        return "Negative"
    return "Neutral"


st.title("Sentiment Analysis App")

email = st.text_input("Enter your Email")
text = st.text_area("Enter your Text")

if st.button("Analyze"):
    if email and text:
        sentiment = predict_sentiment(text)

        data = {
            "email": email,
            "text": text,
            "sentiment": sentiment,
            "timestamp": datetime.datetime.utcnow()
        }

        collection.insert_one(data)

        st.success(f"Sentiment: {sentiment}")
    else:
        st.warning("Please fill all fields")