"""
Transformer NLP Pipeline — FastAPI Application
Author: Adham Aboulkheir
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import time

from models.classifier import TextClassifier
from models.ner import NamedEntityRecogniser


app = FastAPI(
    title="Transformer NLP Pipeline",
    description="End-to-end NLP pipeline: classification, NER, summarisation",
    version="1.0.0"
)

# Initialise models
sentiment_clf = TextClassifier(task="sentiment")
sentiment_clf.fit()

news_clf = TextClassifier(task="news")
news_clf.fit()

ner_model = NamedEntityRecogniser()


class TextRequest(BaseModel):
    text: str
    task: Optional[str] = "sentiment"


class ClassifyRequest(BaseModel):
    text: str


class NERRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "healthy", "models": ["sentiment", "news", "ner"]}


@app.post("/classify/sentiment")
def classify_sentiment(request: ClassifyRequest):
    start = time.time()
    result = sentiment_clf.predict(request.text)
    return {
        "label": result.label,
        "confidence": result.confidence,
        "all_scores": result.all_scores,
        "processing_ms": (time.time() - start) * 1000
    }


@app.post("/classify/news")
def classify_news(request: ClassifyRequest):
    start = time.time()
    result = news_clf.predict(request.text)
    return {
        "label": result.label,
        "confidence": result.confidence,
        "processing_ms": (time.time() - start) * 1000
    }


@app.post("/ner")
def named_entity_recognition(request: NERRequest):
    start = time.time()
    entities = ner_model.predict(request.text)
    return {
        "entities": [{"text": e.text, "label": e.label, "confidence": e.confidence,
                       "start": e.start, "end": e.end} for e in entities],
        "count": len(entities),
        "processing_ms": (time.time() - start) * 1000
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
