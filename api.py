from fastapi import FastAPI
from pydantic import BaseModel, EmailStr
from pymongo import MongoClient
from dotenv import load_dotenv
from urllib.parse import urlparse
import os, datetime

# load env vars
load_dotenv()

# get mongo uri
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise RuntimeError("MONGO_URI not set")

# parse (optional, but fine)
parsed = urlparse(mongo_uri)
db_name = parsed.path[1:]  # sentiment_db

# mongo client
client = MongoClient(mongo_uri)
db = client[db_name]
collection = db["user_sentiments"]

# sanity check
client.admin.command("ping")

# fastapi app
app = FastAPI(title="Sentiment Analysis API")

def predict_sentiment(text: str):
    if "good" in text.lower():
        return "Positive"
    elif "bad" in text.lower():
        return "Negative"
    return "Neutral"

class UserInput(BaseModel):
    email: EmailStr
    text: str

@app.post("/predict")
def predict(user_input: UserInput):
    sentiment = predict_sentiment(user_input.text)

    record = {
        "email": user_input.email,
        "text": user_input.text,
        "sentiment": sentiment,
        "created_at": datetime.datetime.utcnow()
    }

    collection.insert_one(record)

    return {
        "email": user_input.email,
        "sentiment": sentiment
    }
