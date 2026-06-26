"""
Text Classification Model
Author: Adham Aboulkheir
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ClassificationResult:
    text: str
    label: str
    confidence: float
    all_scores: Dict[str, float]


class TextClassifier:
    """
    Text classifier with TF-IDF + Logistic Regression baseline.
    In production: replace with fine-tuned BERT/RoBERTa.
    """
    
    TASK_LABELS = {
        "news":      ["politics", "technology", "sports", "science", "business", "entertainment"],
        "sentiment": ["positive", "negative", "neutral"],
        "intent":    ["question", "command", "statement", "request"],
        "topic":     ["AI", "medicine", "finance", "environment", "education"],
    }
    
    def __init__(self, task: str = "sentiment", model_name: str = "bert-base-uncased"):
        self.task = task
        self.model_name = model_name
        self.labels = self.TASK_LABELS.get(task, ["class_0", "class_1"])
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.label_encoder = LabelEncoder()
        self._fitted = False
    
    def _generate_training_data(self, n_per_class: int = 200):
        """Generate synthetic training data for demonstration."""
        np.random.seed(42)
        texts, labels = [], []
        
        templates = {
            "positive": ["This is excellent and amazing", "Great product highly recommend",
                         "Wonderful experience very happy", "Outstanding quality love it"],
            "negative": ["Terrible experience very disappointed", "Awful quality waste of money",
                         "Horrible service never again", "Worst product ever bought"],
            "neutral":  ["The product arrived on time", "It works as described",
                         "Average quality meets expectations", "Standard delivery nothing special"],
        }
        
        if self.task == "sentiment":
            for label, tmpl_list in templates.items():
                for _ in range(n_per_class):
                    base = np.random.choice(tmpl_list)
                    noise_words = ["really", "quite", "somewhat", "very", "extremely"]
                    text = base + " " + np.random.choice(noise_words)
                    texts.append(text)
                    labels.append(label)
        else:
            for label in self.labels:
                for _ in range(n_per_class):
                    texts.append(f"Sample text about {label} topic with relevant content")
                    labels.append(label)
        
        return texts, labels
    
    def fit(self, texts: List[str] = None, labels: List[str] = None) -> "TextClassifier":
        if texts is None:
            texts, labels = self._generate_training_data()
        
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)
        self.model.fit(X, y)
        self._fitted = True
        return self
    
    def predict(self, text: str) -> ClassificationResult:
        if not self._fitted:
            self.fit()
        
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        classes = self.label_encoder.classes_
        
        pred_idx = np.argmax(proba)
        return ClassificationResult(
            text=text[:100],
            label=classes[pred_idx],
            confidence=float(proba[pred_idx]),
            all_scores={cls: float(p) for cls, p in zip(classes, proba)}
        )
    
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict:
        if not self._fitted:
            self.fit()
        
        X = self.vectorizer.transform(texts)
        y_true = self.label_encoder.transform(labels)
        y_pred = self.model.predict(X)
        
        return {
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "f1_macro":    f1_score(y_true, y_pred, average="macro"),
            "report":      classification_report(y_true, y_pred,
                                                  target_names=self.label_encoder.classes_)
        }


if __name__ == "__main__":
    print("Text Classifier Demo")
    clf = TextClassifier(task="sentiment")
    clf.fit()
    
    test_texts = [
        "This product is absolutely amazing, I love it!",
        "Terrible quality, complete waste of money.",
        "The item arrived on time and works as expected.",
    ]
    
    for text in test_texts:
        result = clf.predict(text)
        print(f"  Text: {text[:50]}...")
        print(f"  Label: {result.label} ({result.confidence:.1%})")
