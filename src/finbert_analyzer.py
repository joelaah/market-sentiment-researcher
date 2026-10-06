from typing import Dict, Any, List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "ProsusAI/finbert"

class FinBERTAnalyzer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FinBERTAnalyzer, cls).__new__(cls)
            cls._instance.tokenizer = None
            cls._instance.model = None
        return cls._instance

    def _load_model(self):
        if self.model is None:
            print(f"[FinBERT] Loading financial sentiment classifier ({MODEL_NAME})...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            self.model.eval()

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyzes financial text and returns pure probabilities:
        - positive: 0.0 to 1.0
        - negative: 0.0 to 1.0
        - neutral: 0.0 to 1.0
        - net_score: positive - negative (bounded strictly between -1.0 and +1.0)
        """
        if not text or not text.strip():
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "net_score": 0.0}

        self._load_model()
        
        # Truncate to maximum 512 tokens
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]

        # FinBERT class labels order: [positive, negative, neutral]
        labels = self.model.config.id2label
        probs_dict = {labels[i].lower(): round(float(probs[i]), 4) for i in range(len(probs))}

        pos = probs_dict.get("positive", 0.0)
        neg = probs_dict.get("negative", 0.0)
        neu = probs_dict.get("neutral", 0.0)
        net_score = round(pos - neg, 4)

        return {
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "net_score": net_score
        }

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, float]]:
        """Batch analyzes multiple articles."""
        return [self.analyze(t) for t in texts]
