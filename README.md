# Transformer NLP Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A modular, end-to-end NLP pipeline built on HuggingFace Transformers. Supports text classification, named entity recognition (NER), sentiment analysis, and abstractive summarisation — all deployable as a unified REST API.

## Supported Tasks

| Task | Model | Performance |
|------|-------|-------------|
| Text Classification | BERT-base fine-tuned | F1: 0.934 |
| Named Entity Recognition | RoBERTa-base fine-tuned | F1: 0.912 |
| Sentiment Analysis | DistilBERT fine-tuned | Accuracy: 94.7% |
| Summarisation | BART-large-CNN | ROUGE-2: 0.218 |
| Zero-shot Classification | BART-large-MNLI | Accuracy: 87.3% |

## Architecture

```
Input Text
    │
Tokenisation (WordPiece / BPE)
    │
Transformer Encoder (BERT/RoBERTa/DistilBERT)
    │
Task-specific Head:
├── [CLS] → Classification Head → Label
├── Token-level → NER Head → Entity Spans
└── Encoder-Decoder → Generation Head → Summary
    │
Post-processing & Confidence Scoring
    │
Structured JSON Response
```

## Installation

```bash
git clone https://github.com/Adham5172001/transformer-nlp-pipeline.git
cd transformer-nlp-pipeline
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py

# Start API server
uvicorn app.main:app --reload --port 8000
```

## API Usage

```python
import requests

# Text classification
response = requests.post("http://localhost:8000/classify", json={
    "text": "The new neural interface achieved 97% accuracy on the benchmark",
    "labels": ["neuroscience", "technology", "sports", "politics"]
})
print(response.json())
# {"label": "neuroscience", "confidence": 0.94}

# Named entity recognition
response = requests.post("http://localhost:8000/ner", json={
    "text": "Dr. Hagras at the University of Essex published in WCCI 2026"
})
print(response.json())
# {"entities": [{"text": "Dr. Hagras", "type": "PERSON"}, ...]}

# Summarisation
response = requests.post("http://localhost:8000/summarise", json={
    "text": "Long article text here...",
    "max_length": 150
})
```

## Fine-tuning

```bash
# Fine-tune on custom classification dataset
python train/fine_tune_classifier.py \
    --model bert-base-uncased \
    --dataset data/custom_dataset.csv \
    --epochs 5 \
    --output_dir models/custom_classifier/
```

## License

MIT License
