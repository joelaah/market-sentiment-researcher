"""
Probability Calibration Module for FinBERT Outputs.

Raw softmax outputs from FinBERT tend to be overconfident (probability mass
concentrated near 0.0 or 1.0). This module provides two calibration methods:

1. **Temperature Scaling**: Divides logits by a learned temperature T > 0 before
   softmax. Simple, preserves class ranking, single scalar parameter.

2. **Platt Scaling (Logistic Calibration)**: Fits a logistic regression on raw
   logits vs. true labels. More flexible but requires per-class fitting.

Usage:
    from src.probability_calibration import TemperatureScaler, PlattCalibrator

    # Temperature Scaling
    scaler = TemperatureScaler()
    scaler.fit(validation_logits, validation_labels)  # numpy arrays
    calibrated = scaler.calibrate_probs(raw_probs)     # dict or array

    # Platt Scaling
    calibrator = PlattCalibrator()
    calibrator.fit(validation_logits, validation_labels)
    calibrated = calibrator.calibrate_probs(raw_probs)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("ProbabilityCalibration")

# ============================================================================
# TEMPERATURE SCALING
# ============================================================================


class TemperatureScaler:
    """
    Learns a single scalar temperature T that minimizes negative log-likelihood
    on a held-out validation set.  At inference time, divides logits by T before
    softmax to produce better-calibrated probabilities.

    If no validation data is available, a default temperature can be set manually.
    """

    DEFAULT_TEMPERATURE = 1.5  # Sensible default that softens overconfident outputs
    CALIBRATION_METHOD = "temperature_scaling"

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Initial temperature value. T=1.0 means no calibration
                         (raw softmax). T>1 softens probabilities, T<1 sharpens.
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature
        self._fitted = False

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over last axis."""
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    @staticmethod
    def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
        """Negative log-likelihood loss."""
        n = len(labels)
        # Clip to avoid log(0)
        eps = 1e-12
        log_probs = np.log(np.clip(probs[np.arange(n), labels], eps, 1.0))
        return -np.mean(log_probs)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 200,
        tol: float = 1e-6,
    ) -> "TemperatureScaler":
        """
        Learns optimal temperature via gradient descent on NLL.

        Args:
            logits: Raw model logits, shape (N, C) where C is number of classes.
            labels: Integer class labels, shape (N,).
            lr: Learning rate for gradient descent.
            max_iter: Maximum optimization iterations.
            tol: Convergence tolerance on temperature change.

        Returns:
            self (fitted).
        """
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        if logits.ndim != 2:
            raise ValueError(f"logits must be 2D (N, C), got shape {logits.shape}")
        if labels.ndim != 1 or len(labels) != len(logits):
            raise ValueError("labels must be 1D with same length as logits")

        T = float(self.temperature)
        n = len(labels)

        for iteration in range(max_iter):
            # Forward: scaled softmax
            scaled_logits = logits / T
            probs = self._softmax(scaled_logits)

            # Gradient of NLL w.r.t. T:
            # dNLL/dT = (1/N) * sum_i [ (p_i - y_i) * logits_i ] * (-1/T^2)
            # Simplified: for each sample, gradient contribution
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(n), labels] = 1.0
            diff = probs - one_hot  # (N, C)

            # dL/dT = mean over samples of sum_c [ diff[i,c] * logits[i,c] * (-1/T^2) ]
            grad = np.mean(np.sum(diff * logits, axis=1)) * (-1.0 / (T * T))

            T_new = T - lr * grad
            # Clamp temperature to reasonable range
            T_new = max(0.1, min(T_new, 20.0))

            if abs(T_new - T) < tol:
                T = T_new
                break
            T = T_new

        self.temperature = T
        self._fitted = True
        logger.info(f"TemperatureScaler fitted: T={self.temperature:.4f}")
        return self

    def set_temperature(self, temperature: float) -> "TemperatureScaler":
        """Manually set temperature without fitting."""
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature
        self._fitted = True
        return self

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to raw logits and return calibrated probabilities."""
        logits = np.asarray(logits, dtype=np.float64)
        scaled = logits / self.temperature
        return self._softmax(scaled)

    def calibrate_probs(self, probs_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Calibrate a FinBERT probability dictionary by reverse-engineering approximate
        logits from the softmax outputs, then re-applying softmax at the learned
        temperature.

        Args:
            probs_dict: Dict with keys 'positive', 'negative', 'neutral' and
                        float probability values.

        Returns:
            New dict with calibrated probabilities and updated net_score/confidence.
        """
        pos = float(probs_dict.get("positive", 0.33))
        neg = float(probs_dict.get("negative", 0.33))
        neu = float(probs_dict.get("neutral", 0.34))

        # Recover approximate logits from probabilities (inverse softmax)
        eps = 1e-12
        raw = np.array([
            math.log(max(pos, eps)),
            math.log(max(neg, eps)),
            math.log(max(neu, eps)),
        ])

        # Apply temperature scaling
        calibrated = self.calibrate_logits(raw.reshape(1, -1))[0]
        cal_pos, cal_neg, cal_neu = float(calibrated[0]), float(calibrated[1]), float(calibrated[2])

        # Reconstruct FinBERT-style output dict
        scores = {"positive": cal_pos, "negative": cal_neg, "neutral": cal_neu}
        polarity = max(scores, key=scores.get)
        confidence = round(scores[polarity], 4)
        net_score = round(cal_pos - cal_neg, 4)

        result = dict(probs_dict)  # preserve any extra keys
        result.update({
            "positive": round(cal_pos, 4),
            "negative": round(cal_neg, 4),
            "neutral": round(cal_neu, 4),
            "net_score": net_score,
            "polarity": polarity,
            "confidence": confidence,
            "calibrated": True,
            "calibration_method": "temperature_scaling",
            "temperature": round(self.temperature, 4),
        })
        return result


# ============================================================================
# PLATT SCALING (LOGISTIC CALIBRATION)
# ============================================================================


class PlattCalibrator:
    """
    Fits per-class logistic regression (Platt scaling) on raw logits vs. binary
    labels for each class. More flexible than temperature scaling but requires
    more validation data.
    """

    CALIBRATION_METHOD = "platt_scaling"

    def __init__(self):
        self._params: Dict[int, Tuple[float, float]] = {}  # class_idx -> (A, B)
        self._fitted = False

    @staticmethod
    def _sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 200,
    ) -> "PlattCalibrator":
        """
        Fits A, B parameters for each class: P(y=c) = sigmoid(A * logit_c + B).

        Args:
            logits: Raw model logits, shape (N, C).
            labels: Integer class labels, shape (N,).
        """
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        n, c = logits.shape

        for cls_idx in range(c):
            # Binary target for this class
            y = (labels == cls_idx).astype(np.float64)
            x = logits[:, cls_idx]

            # Initialize A=1, B=0
            A, B = 1.0, 0.0

            for _ in range(max_iter):
                z = A * x + B
                p = self._sigmoid(z)
                # Gradients
                diff = p - y
                grad_A = np.mean(diff * x)
                grad_B = np.mean(diff)
                A -= lr * grad_A
                B -= lr * grad_B

            self._params[cls_idx] = (A, B)

        self._fitted = True
        logger.info(f"PlattCalibrator fitted for {c} classes")
        return self

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to raw logits and return calibrated probabilities."""
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        n, c = logits.shape
        raw_calibrated = np.zeros_like(logits)

        for cls_idx in range(c):
            A, B = self._params.get(cls_idx, (1.0, 0.0))
            raw_calibrated[:, cls_idx] = self._sigmoid(A * logits[:, cls_idx] + B)

        # Normalize to sum to 1
        row_sums = raw_calibrated.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        return raw_calibrated / row_sums

    def calibrate_probs(self, probs_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Calibrate a FinBERT probability dictionary using Platt scaling.

        Args:
            probs_dict: Dict with 'positive', 'negative', 'neutral' keys.

        Returns:
            Calibrated probability dict.
        """
        pos = float(probs_dict.get("positive", 0.33))
        neg = float(probs_dict.get("negative", 0.33))
        neu = float(probs_dict.get("neutral", 0.34))

        # Recover approximate logits
        eps = 1e-12
        raw = np.array([
            math.log(max(pos, eps)),
            math.log(max(neg, eps)),
            math.log(max(neu, eps)),
        ])

        calibrated = self.calibrate_logits(raw.reshape(1, -1))[0]
        cal_pos, cal_neg, cal_neu = float(calibrated[0]), float(calibrated[1]), float(calibrated[2])

        scores = {"positive": cal_pos, "negative": cal_neg, "neutral": cal_neu}
        polarity = max(scores, key=scores.get)
        confidence = round(scores[polarity], 4)
        net_score = round(cal_pos - cal_neg, 4)

        result = dict(probs_dict)
        result.update({
            "positive": round(cal_pos, 4),
            "negative": round(cal_neg, 4),
            "neutral": round(cal_neu, 4),
            "net_score": net_score,
            "polarity": polarity,
            "confidence": confidence,
            "calibrated": True,
            "calibration_method": "platt_scaling",
        })
        return result


# ============================================================================
# CONVENIENCE: DEFAULT CALIBRATOR FOR FINBERT
# ============================================================================


def get_default_temperature_scaler(temperature: Optional[float] = None) -> TemperatureScaler:
    """
    Returns a TemperatureScaler with a sensible default temperature for FinBERT.

    If no temperature is provided, uses T=1.5 which empirically softens FinBERT's
    overconfident outputs without requiring a validation set.

    Args:
        temperature: Optional explicit temperature. If None, uses default 1.5.

    Returns:
        A ready-to-use TemperatureScaler instance.
    """
    T = temperature if temperature is not None else TemperatureScaler.DEFAULT_TEMPERATURE
    scaler = TemperatureScaler(temperature=T)
    scaler._fitted = True
    return scaler


def calibrate_finbert_output(
    probs_dict: Dict[str, float],
    calibrator: Optional[Union[TemperatureScaler, PlattCalibrator]] = None,
) -> Dict[str, float]:
    """
    Convenience function: apply probability calibration to a single FinBERT output dict.

    If no calibrator is provided, uses the default temperature scaler (T=1.5).

    Args:
        probs_dict: Raw FinBERT output dict with 'positive', 'negative', 'neutral'.
        calibrator: Optional fitted calibrator instance.

    Returns:
        Calibrated probability dict.
    """
    if calibrator is None:
        calibrator = get_default_temperature_scaler()
    return calibrator.calibrate_probs(probs_dict)
