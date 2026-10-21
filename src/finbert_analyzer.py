from typing import Dict, Any, List, Optional
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    HAS_TORCH = False

from src.probability_calibration import (
    TemperatureScaler,
    PlattCalibrator,
    get_default_temperature_scaler,
    calibrate_finbert_output,
)

MODEL_NAME = "ProsusAI/finbert"

class FinBERTAnalyzer:
    _instance = None
    DEFAULT_TEMPERATURE = 1.5

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FinBERTAnalyzer, cls).__new__(cls)
            cls._instance.tokenizer = None
            cls._instance.model = None
            cls._instance.temperature = cls.DEFAULT_TEMPERATURE
            cls._instance.calibrator = None
        return cls._instance

    def __init__(
        self,
        temperature: Optional[float] = None,
        calibrator: Optional[Any] = None,
    ):
        if temperature is not None:
            if temperature <= 0:
                raise ValueError(f"Temperature must be positive, got {temperature}")
            self.temperature = float(temperature)
        elif not hasattr(self, "temperature") or self.temperature is None:
            self.temperature = self.DEFAULT_TEMPERATURE

        if calibrator is not None:
            self.calibrator = calibrator

    def _load_model(self):
        if not HAS_TORCH or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise ImportError(
                "torch and transformers are required to load FinBERT model for inference."
            )
        if self.model is None:
            print(f"[FinBERT] Loading financial sentiment classifier ({MODEL_NAME})...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            self.model.eval()

    def analyze(self, text: str, temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyzes financial text and returns calibrated probabilities and classification:
        - positive: 0.0 to 1.0 (temperature-scaled)
        - negative: 0.0 to 1.0 (temperature-scaled)
        - neutral: 0.0 to 1.0 (temperature-scaled)
        - net_score: positive - negative (bounded strictly between -1.0 and +1.0)
        - polarity: string argmax class ("positive", "negative", or "neutral")
        - confidence: float highest class probability score (rounded to 4 decimal places)
        - calibrated: True
        - temperature: float temperature used
        """
        T = float(temperature if temperature is not None else self.temperature)
        if T <= 0:
            raise ValueError(f"Temperature must be positive, got {T}")

        if not text or not text.strip():
            return {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "net_score": 0.0,
                "polarity": "neutral",
                "confidence": 1.0,
                "calibrated": True,
                "temperature": T,
            }

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
            # If an external calibrator is supplied with calibrate_logits, apply directly
            if self.calibrator is not None and hasattr(self.calibrator, "calibrate_logits"):
                raw_logits_np = outputs.logits.detach().cpu().numpy()
                probs = self.calibrator.calibrate_logits(raw_logits_np)[0]
                cal_method = getattr(self.calibrator, "CALIBRATION_METHOD", "platt_scaling")
            else:
                # Temperature scaling on raw logits before softmax
                scaled_logits = outputs.logits / T
                probs = F.softmax(scaled_logits, dim=-1)[0]
                cal_method = "temperature_scaling"

        # FinBERT class labels order: [positive, negative, neutral]
        labels = self.model.config.id2label
        probs_dict = {labels[i].lower(): round(float(probs[i]), 4) for i in range(len(probs))}

        pos = probs_dict.get("positive", 0.0)
        neg = probs_dict.get("negative", 0.0)
        neu = probs_dict.get("neutral", 0.0)
        net_score = round(pos - neg, 4)

        scores = {"positive": pos, "negative": neg, "neutral": neu}
        polarity = max(scores, key=scores.get)
        confidence = round(float(scores[polarity]), 4)

        res = {
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "net_score": net_score,
            "polarity": polarity,
            "confidence": confidence,
            "calibrated": True,
            "calibration_method": cal_method,
            "temperature": T,
        }

        return res

    def batch_analyze(self, texts: List[str], temperature: Optional[float] = None) -> List[Dict[str, Any]]:
        """Batch analyzes multiple articles."""
        return [self.analyze(t, temperature=temperature) for t in texts]
