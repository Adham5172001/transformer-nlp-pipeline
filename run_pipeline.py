"""
Transformer NLP Pipeline — Full Demo
Author: Adham Aboulkheir
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from models.classifier import TextClassifier
from models.ner import NamedEntityRecogniser


def main():
    print("=" * 55)
    print("TRANSFORMER NLP PIPELINE DEMO")
    print("Author: Adham Aboulkheir")
    print("=" * 55)
    
    print("\n[1/3] Sentiment Analysis")
    clf = TextClassifier(task="sentiment")
    clf.fit()
    
    texts = [
        "This product is absolutely amazing, I love it!",
        "Terrible quality, complete waste of money.",
        "The item arrived on time and works as expected.",
        "The research paper presents interesting findings.",
    ]
    
    for text in texts:
        result = clf.predict(text)
        print(f"  {result.label.upper():10} ({result.confidence:.0%}) | {text[:50]}...")
    
    print("\n[2/3] Named Entity Recognition")
    ner = NamedEntityRecogniser()
    
    test_text = "Dr. Adham Aboulkheir from the University of Essex published research on fuzzy logic and MEA neural biocomputers at WCCI 2026 in Maastricht, achieving 97.74% F1-score."
    entities = ner.predict(test_text)
    print(f"  Text: {test_text[:70]}...")
    print(f"  Entities ({len(entities)} found):")
    for e in entities[:6]:
        print(f"    [{e.label}] {e.text!r}")
    
    print("\n[3/3] Multi-task Pipeline")
    pipeline_texts = [
        "OpenAI released GPT-4 in March 2023, achieving 90% on the bar exam.",
        "The University of Essex research team published in IEEE FUZZ 2026.",
        "BT Group reported 15% revenue growth in Q3 2025.",
    ]
    
    for text in pipeline_texts:
        sentiment = clf.predict(text)
        entities = ner.predict(text)
        print(f"\n  Text: {text[:60]}...")
        print(f"  Sentiment: {sentiment.label} ({sentiment.confidence:.0%})")
        print(f"  Entities: {[e.text for e in entities[:3]]}")
    
    print("\n✓ Pipeline demo complete!")


if __name__ == "__main__":
    main()
