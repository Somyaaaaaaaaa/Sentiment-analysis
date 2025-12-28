import re
import torch
import torch.nn.functional as F
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AutoTokenizer,
    AutoModel
)

# ================== LOAD MODELS ==================

# Valence model (sentiment)
sent_tokenizer = DistilBertTokenizerFast.from_pretrained("sentiment_bert")
sent_model = DistilBertForSequenceClassification.from_pretrained("sentiment_bert")
sent_model.eval()

# Embedding model (semantic emotion)
embed_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
embed_model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
embed_model.eval()

# ================== INTENSITY ==================

PROFANITY = {"fuck", "fucking", "shit", "damn", "hell"}

def detect_intensity(text: str):
    score = 0
    t = text.lower()

    if any(p in t for p in PROFANITY):
        score += 1
    if "!" in text:
        score += 1
    if re.search(r"(very|so|extremely|really)", t):
        score += 1

    if score >= 2:
        return "high"
    elif score == 1:
        return "medium"
    else:
        return "low"

# ================== SEVERITY ==================

DISTRESS_PHRASES = [
    "i can't do this anymore",
    "i cant do this anymore",
    "i'm done",
    "im done",
    "i am done",
    "i don't know how to go on",
    "i dont know how to go on",
    "i'm so tired of everything",
    "im so tired of everything",
    "i am depressed",
    "i feel depressed"
]

def detect_severity(text: str):
    t = text.lower()
    if any(p in t for p in DISTRESS_PHRASES) or "suicidal" in t:
        return "high"
    return "normal"

# ================== FIRST-PERSON OVERRIDES ==================

FIRST_PERSON_EMOTIONS = {
    "i am sad": "sadness",
    "i feel sad": "sadness",
    "i'm sad": "sadness",
    "im sad": "sadness"
}

# ================== EMBEDDINGS ==================

def get_embedding(text):
    inputs = embed_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# ================== EMOTION PROTOTYPES ==================

EMOTION_PROTOTYPES = {
    "sadness": "I feel sad, low, and down",
    "hopelessness": "I feel empty and see no way forward",
    "distress": "I feel overwhelmed and unable to cope",
    "anxiety": "I feel worried and tense about what might happen",
    "loneliness": "I feel alone and disconnected",
    "anger": "I feel angry, irritated, and frustrated",
    "relief": "I feel lighter and can breathe again",
    "gratitude": "I feel thankful and appreciative"
}

EMOTION_VALENCE = {
    "sadness": "negative",
    "hopelessness": "negative",
    "distress": "negative",
    "anxiety": "negative",
    "loneliness": "negative",
    "anger": "negative",
    "relief": "positive",
    "gratitude": "positive"
}

# ================== EMOTION DETECTION ==================

def detect_emotion(text: str, intensity: str):
    text_emb = get_embedding(text)

    scores = {}
    for emotion, proto in EMOTION_PROTOTYPES.items():
        proto_emb = get_embedding(proto)
        score = F.cosine_similarity(text_emb, proto_emb).item()
        scores[emotion] = score

    best_emotion, best_score = max(scores.items(), key=lambda x: x[1])

    if best_score < 0.35:
        return "uncertain"

    # Guardrail: flat language should not become anger
    if best_emotion == "anger" and intensity == "low":
        return "sadness"

    return best_emotion

# ================== MAIN ANALYSIS ==================

def analyze_text(text: str):
    t = text.lower()

    # ----- Valence -----
    inputs = sent_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        logits = sent_model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0]
    neg, pos = probs.tolist()

    if pos > 0.6:
        valence = "positive"
        confidence = pos
    elif neg > 0.6:
        valence = "negative"
        confidence = neg
    else:
        valence = "mixed"
        confidence = max(pos, neg)

    intensity = detect_intensity(text)
    severity = detect_severity(text)

    # ----- FIRST-PERSON OVERRIDE (HIGHEST PRIORITY) -----
    emotion = None
    for phrase, emo in FIRST_PERSON_EMOTIONS.items():
        if phrase in t:
            emotion = emo
            valence = "negative"   # override IMDb bias
            break

    # ----- FALLBACK TO SEMANTIC DETECTION -----
    if emotion is None:
        emotion = detect_emotion(text, intensity)

    # ----- EMOTION–VALENCE CONSISTENCY -----
    if emotion in EMOTION_VALENCE:
        if EMOTION_VALENCE[emotion] != valence:
            emotion = "uncertain"

    return {
        "valence": valence,
        "emotion": emotion,
        "intensity": intensity,
        "severity": severity,
        "confidence": round(confidence, 2)
    }


