from fastapi import FastAPI
from pydantic import BaseModel
from analyzer import analyze_text
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os

# -------- ENV --------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI missing")

# -------- DB --------
client = MongoClient(MONGO_URI)
db = client["sentiment_db"]
collection = db["user_inputs"]

# -------- APP --------
app = FastAPI(title="Emotional Signal API")

class AnalyzeRequest(BaseModel):
    text: str
    email: str | None = None

class AnalyzeResponse(BaseModel):
    valence: str
    emotion: str
    intensity: str
    severity: str
    confidence: float

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(data: AnalyzeRequest):
    result = analyze_text(data.text)

    collection.insert_one({
        "email": data.email,
        "text": data.text,
        "result": result,
        "timestamp": datetime.utcnow()
    })

    return result
