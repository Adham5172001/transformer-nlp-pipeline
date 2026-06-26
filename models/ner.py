"""
Named Entity Recognition Model
Author: Adham Aboulkheir
"""
import re
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    confidence: float


class NamedEntityRecogniser:
    """
    Rule-based + ML NER model.
    In production: replace with fine-tuned RoBERTa-NER.
    """
    
    PATTERNS = {
        "PERSON":  [r"\b(?:Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)?\s*[A-Z][a-z]+ [A-Z][a-z]+\b"],
        "ORG":     [r"\b(?:University of|BT Group|Saudi Motorsport|ThresholdXpert|IEEE|ETH Zurich|OpenAI|Google|Microsoft|Amazon)\b"],
        "LOC":     [r"\b(?:United Kingdom|Essex|Maastricht|London|Egypt|Netherlands|USA|Germany|France)\b"],
        "DATE":    [r"\b(?:20[0-9]{2}|January|February|March|April|May|June|July|August|September|October|November|December)\b"],
        "TECH":    [r"\b(?:BERT|GPT-4|LLM|RAG|GAN|VAE|CNN|LSTM|XAI|MEA|fuzzy logic|neural network|machine learning|deep learning)\b"],
        "METRIC":  [r"\b\d+\.?\d*\s*(?:%|percent|Hz|ms|GB|TB|km/h|°C)\b"],
    }
    
    def __init__(self, model_name: str = "roberta-base-ner"):
        self.model_name = model_name
        self._compiled = {
            label: [re.compile(p, re.IGNORECASE) for p in patterns]
            for label, patterns in self.PATTERNS.items()
        }
    
    def predict(self, text: str) -> List[Entity]:
        """Extract named entities from text."""
        entities = []
        seen_spans = set()
        
        for label, patterns in self._compiled.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    span = (match.start(), match.end())
                    if span not in seen_spans:
                        seen_spans.add(span)
                        entities.append(Entity(
                            text=match.group().strip(),
                            label=label,
                            start=match.start(),
                            end=match.end(),
                            confidence=round(0.85 + 0.1 * (len(match.group()) > 5), 3)
                        ))
        
        return sorted(entities, key=lambda e: e.start)
    
    def batch_predict(self, texts: List[str]) -> List[List[Entity]]:
        return [self.predict(text) for text in texts]


if __name__ == "__main__":
    print("NER Demo")
    ner = NamedEntityRecogniser()
    
    text = "Dr. Adham Aboulkheir from the University of Essex published a paper on fuzzy logic for neural biocomputers at WCCI FUZZ-IEEE 2026 in Maastricht. The research achieved 97.74% F1-score using XAI techniques with MEA data from ETH Zurich."
    
    entities = ner.predict(text)
    print(f"Text: {text[:80]}...")
    print(f"\nEntities found: {len(entities)}")
    for e in entities:
        print(f"  [{e.label}] {e.text!r} ({e.confidence:.0%})")
