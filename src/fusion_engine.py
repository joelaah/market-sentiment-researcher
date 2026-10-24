"""
Multi-Modal Fusion Engine with 3-Tier Fallback Architecture.

Combines calibrated FinBERT sentiment logits, BGE-Small dense text components (384-d),
and quantitative technical indicators (14-period RSI, 5-day momentum, 20-day Volume Z-Score,
14-period ATR volatility) to predict forward trade edge probability P(edge) in [0.0, 1.0].

Hierarchy:
  - Tier 1: LightGBM binary classifier with monotonic constraints (+1 on net sentiment).
  - Tier 2: Scikit-learn calibrated logistic regression model.
  - Tier 3: Pure NumPy fallback (NumpyFusionFallback) with strictly positive sentiment weight (w_sent > 0),
            mathematically guaranteeing that higher net sentiment strictly increases trade edge probability
            when technical factors are held constant.
"""

import logging
import math
import os
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    pa = None
    pq = None
    HAS_PYARROW = False

logger = logging.getLogger("MultiModalFusionEngine")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEEDBACK_PARQUET_FILE = DATA_DIR / "trade_feedback.parquet"
FEEDBACK_JSON_FILE = DATA_DIR / "trade_feedback.json"


# Optional Tier 1: LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, Exception):
    lgb = None
    HAS_LIGHTGBM = False

# Optional Tier 2: Scikit-Learn
try:
    import sklearn
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except (ImportError, Exception):
    sklearn = None
    LogisticRegression = None
    HAS_SKLEARN = False

# Optional FinBERT & Calibration integration
try:
    from src.finbert_analyzer import FinBERTAnalyzer
except ImportError:
    FinBERTAnalyzer = None

try:
    from src.probability_calibration import (
        TemperatureScaler,
        PlattCalibrator,
        calibrate_finbert_output,
        get_default_temperature_scaler,
    )
except ImportError:
    TemperatureScaler = None
    PlattCalibrator = None
    calibrate_finbert_output = None
    get_default_temperature_scaler = None


# Canonical feature schema
FEATURE_NAMES = [
    "net_sentiment",          # 0: Net sentiment polarity in [-1.0, 1.0]
    "positive_prob",          # 1: Calibrated positive probability in [0.0, 1.0]
    "negative_prob",          # 2: Calibrated negative probability in [0.0, 1.0]
    "neutral_prob",           # 3: Calibrated neutral probability in [0.0, 1.0]
    "sentiment_logit_pos",    # 4: Positive logit
    "sentiment_logit_neg",    # 5: Negative logit
    "sentiment_logit_neu",    # 6: Neutral logit
    "rsi_14",                 # 7: 14-period RSI in [0.0, 100.0]
    "momentum_5d",            # 8: 5-day percentage momentum in [-25.0, 25.0]
    "volume_z_score",         # 9: 20-day volume Z-score in [-3.0, 3.0]
    "atr_volatility",         # 10: 14-period normalized ATR volatility % in [0.0, 20.0]
    "above_sma20",            # 11: 1.0 if price >= SMA20, else 0.0
    "catalyst_clarity",       # 12: Catalyst clarity score in [0.05, 1.00]
    "text_anchor_similarity", # 13: Cosine similarity with catalyst anchor in [-1.0, 1.0]
]

# Monotone constraints for LightGBM (+1: increasing, -1: decreasing, 0: unconstrained)
# Index 0: net_sentiment -> +1 (strictly monotonic increasing)
# Index 1: positive_prob -> +1 (monotonic increasing)
# Index 2: negative_prob -> -1 (monotonic decreasing)
LIGHTGBM_MONOTONE_CONSTRAINTS = [1, 1, -1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0]

# Deterministic reference catalyst anchor vector (384 dims, unit normalized)
_SEED_RNG = np.random.RandomState(42)
_RAW_CATALYST_ANCHOR = _SEED_RNG.normal(loc=0.0, scale=1.0, size=(384,)).astype(np.float64)
CATALYST_ANCHOR_VECTOR = _RAW_CATALYST_ANCHOR / (np.linalg.norm(_RAW_CATALYST_ANCHOR) + 1e-12)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts input to float, falling back on default if None or NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


# ============================================================================
# TIER 3: PURE NUMPY CALIBRATED LOGISTIC FALLBACK
# ============================================================================

class NumpyFusionFallback:
    """
    Pure NumPy implementation of multi-modal fusion scoring.
    Zero external ML library dependencies.

    Mathematical Monotonicity Guarantee:
    Let the logit be:
        z = w_0 + w_sent * net_sentiment + w_pos * positive_prob - w_neg * negative_prob
            + w_rsi * ((rsi - 50) / 25) + w_mom * (momentum_5d / 5)
            + w_vol * volume_z_score + w_above * above_sma20
            - w_atr * atr_volatility + w_clarity * catalyst_clarity
            + w_anchor * text_anchor_similarity + text_dense_projection

    Then P(edge) = sigma(z) = 1 / (1 + exp(-z)).
    Since d sigma(z) / dz = sigma(z) * (1 - sigma(z)) > 0 for all finite z,
    and dz / d(net_sentiment) = w_sent > 0,
    the partial derivative:
        dP / d(net_sentiment) = w_sent * sigma(z) * (1 - sigma(z)) > 0
    is strictly positive everywhere, mathematically guaranteeing that higher
    net sentiment strictly increases trade edge probability when technical
    factors are held constant.
    """

    def __init__(
        self,
        w_sent: float = 1.35,
        w_pos: float = 0.45,
        w_neg: float = 0.25,
        w_rsi: float = 0.40,
        w_mom: float = 0.08,
        w_vol: float = 0.12,
        w_above: float = 0.20,
        w_atr: float = 0.04,
        w_clarity: float = 0.35,
        w_anchor: float = 0.25,
        w_0: float = -0.15
    ):
        assert w_sent > 0, "w_sent must be strictly positive for monotonic guarantee"
        self.w_sent = w_sent
        self.w_pos = w_pos
        self.w_neg = w_neg
        self.w_rsi = w_rsi
        self.w_mom = w_mom
        self.w_vol = w_vol
        self.w_above = w_above
        self.w_atr = w_atr
        self.w_clarity = w_clarity
        self.w_anchor = w_anchor
        self.w_0 = w_0

    @staticmethod
    def _sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Numerically stable sigmoid function."""
        z_clipped = np.clip(z, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def compute_logit(
        self,
        features: Dict[str, Any],
        dense_projection: float = 0.0
    ) -> float:
        """Computes the calibrated latent logit from features."""
        net = float(np.clip(features.get("net_sentiment", 0.0), -1.0, 1.0))
        pos = float(np.clip(features.get("positive_prob", 0.33), 0.0, 1.0))
        neg = float(np.clip(features.get("negative_prob", 0.33), 0.0, 1.0))

        # Technical factors
        rsi = float(np.clip(features.get("rsi_14", 50.0), 0.0, 100.0))
        norm_rsi = (rsi - 50.0) / 25.0

        mom_5d = float(np.clip(features.get("momentum_5d", 0.0), -20.0, 20.0))
        norm_mom = mom_5d / 5.0

        vol_z = float(np.clip(features.get("volume_z_score", 0.0), -3.0, 3.0))
        above = 1.0 if features.get("above_sma20", False) else 0.0
        atr = float(np.clip(features.get("atr_volatility", 0.0), 0.0, 15.0))

        # Text & catalyst factors
        clarity = float(np.clip(features.get("catalyst_clarity", 0.10), 0.05, 1.0))
        anchor_sim = float(np.clip(features.get("text_anchor_similarity", 0.0), -1.0, 1.0))

        z = (
            self.w_0
            + (self.w_sent * net)
            + (self.w_pos * pos)
            - (self.w_neg * neg)
            + (self.w_rsi * norm_rsi)
            + (self.w_mom * norm_mom)
            + (self.w_vol * vol_z)
            + (self.w_above * above)
            - (self.w_atr * atr)
            + (self.w_clarity * clarity)
            + (self.w_anchor * anchor_sim)
            + (0.15 * dense_projection)
        )
        return float(z)

    def predict_proba(
        self,
        features: Dict[str, Any],
        dense_projection: float = 0.0
    ) -> float:
        """Returns continuous edge probability in [0.0, 1.0]."""
        z = self.compute_logit(features, dense_projection=dense_projection)
        prob = float(self._sigmoid(z))
        return float(np.clip(prob, 0.0, 1.0))

    def update_from_feedback(
        self,
        feedback_samples: List[Tuple[Dict[str, Any], int]],
        learning_rate: float = 0.05,
        loss_penalty_multiplier: float = 1.5,
        epochs: int = 5,
    ) -> Dict[str, Any]:
        """
        Updates weights via asymmetric error gradient descent on trade feedback.
        Penalizes feature combinations that led to stopped-out trades (y=0) more heavily
        than rewarding winning trades (y=1).
        Preserves the strict monotonicity invariant: w_sent >= 0.10.
        """
        if not feedback_samples:
            return {"status": "no_samples", "samples": 0}

        initial_w = {
            "w_sent": self.w_sent, "w_pos": self.w_pos, "w_neg": self.w_neg,
            "w_rsi": self.w_rsi, "w_mom": self.w_mom, "w_vol": self.w_vol,
            "w_above": self.w_above, "w_atr": self.w_atr, "w_clarity": self.w_clarity,
            "w_anchor": self.w_anchor, "w_0": self.w_0
        }

        for _ in range(epochs):
            for features, label in feedback_samples:
                y = float(label)
                z = self.compute_logit(features)
                p = float(self._sigmoid(z))

                # Asymmetric mistake penalty: mistakes (y=0) are scaled by loss_penalty_multiplier
                err = p - y
                if y == 0.0:
                    err *= float(loss_penalty_multiplier)

                # Feature values
                net = float(np.clip(features.get("net_sentiment", 0.0), -1.0, 1.0))
                pos = float(np.clip(features.get("positive_prob", 0.33), 0.0, 1.0))
                neg = float(np.clip(features.get("negative_prob", 0.33), 0.0, 1.0))
                rsi = float(np.clip(features.get("rsi_14", 50.0), 0.0, 100.0))
                norm_rsi = (rsi - 50.0) / 25.0
                mom = float(np.clip(features.get("momentum_5d", 0.0), -20.0, 20.0))
                norm_mom = mom / 5.0
                vol_z = float(np.clip(features.get("volume_z_score", 0.0), -3.0, 3.0))
                above = 1.0 if features.get("above_sma20", False) else 0.0
                atr = float(np.clip(features.get("atr_volatility", 0.0), 0.0, 15.0))
                clarity = float(np.clip(features.get("catalyst_clarity", 0.10), 0.05, 1.0))
                anchor = float(np.clip(features.get("text_anchor_similarity", 0.0), -1.0, 1.0))

                step = learning_rate * err

                # Weight updates
                self.w_0 -= step
                self.w_sent -= step * net
                self.w_pos -= step * pos
                self.w_neg += step * neg
                self.w_rsi -= step * norm_rsi
                self.w_mom -= step * norm_mom
                self.w_vol -= step * vol_z
                self.w_above -= step * above
                self.w_atr += step * atr
                self.w_clarity -= step * clarity
                self.w_anchor -= step * anchor

                # Monotonicity & stability guards
                self.w_sent = float(max(0.10, min(5.0, self.w_sent)))
                self.w_pos = float(max(0.0, min(3.0, self.w_pos)))
                self.w_neg = float(max(0.0, min(3.0, self.w_neg)))
                self.w_rsi = float(np.clip(self.w_rsi, -2.0, 2.0))
                self.w_mom = float(np.clip(self.w_mom, -2.0, 2.0))
                self.w_vol = float(np.clip(self.w_vol, -2.0, 2.0))
                self.w_above = float(np.clip(self.w_above, -2.0, 2.0))
                self.w_atr = float(np.clip(self.w_atr, 0.0, 2.0))
                self.w_clarity = float(np.clip(self.w_clarity, 0.0, 2.0))
                self.w_anchor = float(np.clip(self.w_anchor, -2.0, 2.0))
                self.w_0 = float(np.clip(self.w_0, -3.0, 3.0))

        updated_w = {
            "w_sent": self.w_sent, "w_pos": self.w_pos, "w_neg": self.w_neg,
            "w_rsi": self.w_rsi, "w_mom": self.w_mom, "w_vol": self.w_vol,
            "w_above": self.w_above, "w_atr": self.w_atr, "w_clarity": self.w_clarity,
            "w_anchor": self.w_anchor, "w_0": self.w_0
        }

        return {
            "status": "updated",
            "samples": len(feedback_samples),
            "initial_weights": initial_w,
            "updated_weights": updated_w
        }



# ============================================================================
# TIER 2: SCIKIT-LEARN LOGISTIC REGRESSION MODEL
# ============================================================================

class SklearnFusionModel:
    """
    Tier 2: Scikit-learn calibrated logistic regression model.
    Utilizes positive coefficients on sentiment features to enforce monotonic behavior.
    """

    def __init__(self):
        self.model: Optional[Any] = None
        self._fitted = False
        self._baseline_X: Optional[np.ndarray] = None
        self._baseline_y: Optional[np.ndarray] = None
        self._initialize_default_model()

    def _initialize_default_model(self):
        if not HAS_SKLEARN:
            return
        try:
            # Synthetic calibration dataset capturing joint distribution
            rng = np.random.RandomState(1337)
            n_samples = 600

            # Generate synthetic feature rows
            net_sent = rng.uniform(-1.0, 1.0, size=(n_samples, 1))
            pos_prob = np.clip((net_sent + 1.0) / 2.0 + rng.normal(0, 0.05, (n_samples, 1)), 0.0, 1.0)
            neg_prob = np.clip(1.0 - pos_prob, 0.0, 1.0)
            neu_prob = np.clip(1.0 - (pos_prob + neg_prob), 0.0, 1.0)

            # Logits
            logit_pos = np.log(np.clip(pos_prob, 1e-4, 1.0))
            logit_neg = np.log(np.clip(neg_prob, 1e-4, 1.0))
            logit_neu = np.log(np.clip(neu_prob, 1e-4, 1.0))

            rsi_14 = rng.uniform(25.0, 75.0, size=(n_samples, 1))
            mom_5d = rng.uniform(-8.0, 8.0, size=(n_samples, 1))
            vol_z = rng.uniform(-2.0, 2.0, size=(n_samples, 1))
            atr = rng.uniform(1.0, 5.0, size=(n_samples, 1))
            above_sma = (mom_5d >= 0.0).astype(float)
            clarity = rng.uniform(0.1, 0.9, size=(n_samples, 1))
            anchor_sim = rng.uniform(-0.5, 0.8, size=(n_samples, 1))

            X = np.hstack([
                net_sent, pos_prob, neg_prob, neu_prob,
                logit_pos, logit_neg, logit_neu,
                rsi_14, mom_5d, vol_z, atr, above_sma,
                clarity, anchor_sim
            ])

            # Label generation: edge probability target
            latent = (
                1.2 * net_sent
                + 0.4 * pos_prob
                - 0.3 * neg_prob
                + 0.02 * (rsi_14 - 50.0)
                + 0.08 * mom_5d
                + 0.10 * vol_z
                + 0.15 * above_sma
                - 0.05 * atr
                + 0.30 * clarity
                + 0.20 * anchor_sim
            )
            probs = 1.0 / (1.0 + np.exp(-latent))
            y = (probs >= 0.50).astype(int).ravel()

            clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
            clf.fit(X, y)
            self.model = clf
            self._fitted = True
            self._baseline_X = X
            self._baseline_y = y
        except Exception as e:
            logger.warning(f"Could not initialize SklearnFusionModel: {e}")
            self.model = None
            self._fitted = False

    def update_with_feedback(
        self,
        X_feedback: np.ndarray,
        y_feedback: np.ndarray,
        loss_penalty_multiplier: float = 1.5,
    ) -> bool:
        """
        Incorporates feedback via experience replay:
        Combines baseline calibration distribution with feedback samples
        to prevent catastrophic forgetting and maintain sentiment monotonicity.
        """
        if not self.is_available() or self._baseline_X is None or self._baseline_y is None:
            return False
        try:
            weights_fb = np.where(y_feedback == 0, float(loss_penalty_multiplier), 1.0)
            X_combined = np.vstack([self._baseline_X, X_feedback])
            y_combined = np.concatenate([self._baseline_y, y_feedback])
            w_combined = np.concatenate([np.ones(len(self._baseline_y)), weights_fb])

            clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
            clf.fit(X_combined, y_combined, sample_weight=w_combined)
            # Ensure sentiment coefficient preserves positive monotonicity
            if hasattr(clf, "coef_") and clf.coef_[0, 0] > 0.05:
                self.model = clf
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to update SklearnFusionModel with feedback: {e}")
            return False

    def is_available(self) -> bool:
        return HAS_SKLEARN and self.model is not None and self._fitted

    def predict_proba(self, feature_vector: np.ndarray) -> float:
        """Predicts edge probability from feature vector (14 dims)."""
        if not self.is_available():
            raise RuntimeError("SklearnFusionModel is not available or not fitted")
        X = feature_vector.reshape(1, -1)
        prob = float(self.model.predict_proba(X)[0, 1])
        return float(np.clip(prob, 0.0, 1.0))


# ============================================================================
# TIER 1: LIGHTGBM BINARY CLASSIFIER WITH MONOTONIC CONSTRAINTS
# ============================================================================

class LightGBMFusionModel:
    """
    Tier 1: LightGBM binary classifier with monotonic constraints.
    Enforces monotone_constraints = +1 on net sentiment, guaranteeing:
        dP / d(net_sentiment) >= 0.
    """

    def __init__(self, model_file: Optional[str] = None):
        self.booster: Optional[Any] = None
        self._model_file = model_file or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "fusion_lgbm.txt"
        )
        self._initialize_booster()

    def _initialize_booster(self):
        if not HAS_LIGHTGBM:
            return
        try:
            # 1. Try loading cached model from file
            if os.path.exists(self._model_file):
                try:
                    self.booster = lgb.Booster(model_file=self._model_file)
                    return
                except Exception as e:
                    logger.warning(f"Error loading LightGBM booster from {self._model_file}: {e}")

            # 2. Train and save calibrated booster with monotonic constraints
            rng = np.random.RandomState(2026)
            n_samples = 1200

            net_sent = rng.uniform(-1.0, 1.0, size=(n_samples, 1))
            pos_prob = np.clip((net_sent + 1.0) / 2.0 + rng.normal(0, 0.04, (n_samples, 1)), 0.0, 1.0)
            neg_prob = np.clip(1.0 - pos_prob, 0.0, 1.0)
            neu_prob = np.clip(1.0 - (pos_prob + neg_prob), 0.0, 1.0)

            logit_pos = np.log(np.clip(pos_prob, 1e-4, 1.0))
            logit_neg = np.log(np.clip(neg_prob, 1e-4, 1.0))
            logit_neu = np.log(np.clip(neu_prob, 1e-4, 1.0))

            rsi_14 = rng.uniform(20.0, 80.0, size=(n_samples, 1))
            mom_5d = rng.uniform(-10.0, 10.0, size=(n_samples, 1))
            vol_z = rng.uniform(-3.0, 3.0, size=(n_samples, 1))
            atr = rng.uniform(0.5, 6.0, size=(n_samples, 1))
            above_sma = (mom_5d >= 0.0).astype(float)
            clarity = rng.uniform(0.1, 1.0, size=(n_samples, 1))
            anchor_sim = rng.uniform(-0.6, 0.9, size=(n_samples, 1))

            X = np.hstack([
                net_sent, pos_prob, neg_prob, neu_prob,
                logit_pos, logit_neg, logit_neu,
                rsi_14, mom_5d, vol_z, atr, above_sma,
                clarity, anchor_sim
            ])

            latent = (
                1.4 * net_sent
                + 0.5 * pos_prob
                - 0.3 * neg_prob
                + 0.025 * (rsi_14 - 50.0)
                + 0.10 * mom_5d
                + 0.12 * vol_z
                + 0.20 * above_sma
                - 0.04 * atr
                + 0.35 * clarity
                + 0.25 * anchor_sim
            )
            prob_targets = 1.0 / (1.0 + np.exp(-latent))
            y = (prob_targets >= 0.50).astype(int).ravel()

            train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES, free_raw_data=False)
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "monotone_constraints": LIGHTGBM_MONOTONE_CONSTRAINTS,
                "learning_rate": 0.05,
                "num_leaves": 15,
                "verbose": -1,
                "min_child_samples": 20,
                "seed": 42
            }
            self.booster = lgb.train(params, train_data, num_boost_round=60)

            # Ensure parent data directory exists and save
            os.makedirs(os.path.dirname(self._model_file), exist_ok=True)
            self.booster.save_model(self._model_file)
        except Exception as e:
            logger.warning(f"Could not initialize LightGBMFusionModel: {e}")
            self.booster = None

    def is_available(self) -> bool:
        return HAS_LIGHTGBM and self.booster is not None

    def predict_proba(self, feature_vector: np.ndarray) -> float:
        """Predicts calibrated probability using LightGBM booster."""
        if not self.is_available():
            raise RuntimeError("LightGBMFusionModel is not available")
        X = feature_vector.reshape(1, -1)
        prob = float(self.booster.predict(X)[0])
        return float(np.clip(prob, 0.0, 1.0))


# ============================================================================
# ORCHESTRATOR: MULTI-MODAL FUSION ENGINE
# ============================================================================

class MultiModalFusionEngine:
    """
    Learned Multi-Modal Fusion Engine.
    Coordinates feature extraction and executes the highest available tier:
      Tier 1: LightGBM (monotone constraints on sentiment)
      Tier 2: Scikit-Learn
      Tier 3: Pure NumPy Fallback (mathematically monotonic w_sent > 0)
    """

    def __init__(
        self,
        tier: Optional[int] = None,
        finbert_analyzer: Optional[Any] = None,
        calibrator: Optional[Any] = None,
    ):
        """
        Args:
            tier: Optional tier override (1, 2, or 3). If None, auto-selects highest available.
            finbert_analyzer: Optional injected FinBERTAnalyzer singleton instance.
            calibrator: Optional injected probability calibrator (TemperatureScaler or PlattCalibrator).
        """
        self.tier_override = tier
        self.finbert_analyzer = finbert_analyzer
        self.calibrator = calibrator
        self.lgbm_model: Optional[LightGBMFusionModel] = None
        self.sklearn_model: Optional[SklearnFusionModel] = None
        self.numpy_fallback = NumpyFusionFallback()

        # Initialize available models
        if tier == 1 or (tier is None and HAS_LIGHTGBM):
            try:
                self.lgbm_model = LightGBMFusionModel()
            except Exception as e:
                logger.warning(f"LightGBM initialization failed: {e}")

        if tier == 2 or (tier is None and HAS_SKLEARN):
            try:
                self.sklearn_model = SklearnFusionModel()
            except Exception as e:
                logger.warning(f"Sklearn initialization failed: {e}")

    @property
    def active_tier(self) -> str:
        """Returns description of the currently active backend tier."""
        if self.tier_override == 1:
            if self.lgbm_model and self.lgbm_model.is_available():
                return "Tier 1: LightGBM"
            return "Tier 1: LightGBM (Unavailable)"
        if self.tier_override == 2:
            if self.sklearn_model and self.sklearn_model.is_available():
                return "Tier 2: Scikit-Learn"
            return "Tier 2: Scikit-Learn (Unavailable)"
        if self.tier_override == 3:
            return "Tier 3: Pure NumPy Fallback"

        # Auto-selection
        if self.lgbm_model and self.lgbm_model.is_available():
            return "Tier 1: LightGBM"
        if self.sklearn_model and self.sklearn_model.is_available():
            return "Tier 2: Scikit-Learn"
        return "Tier 3: Pure NumPy Fallback"

    def extract_features(
        self,
        features: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], np.ndarray, float]:
        """
        Extracts, normalizes, and sanitizes multi-modal features.

        Returns:
            Tuple of:
              - features_dict: clean dictionary of extracted scalar metrics
              - feature_vector: 14-dimensional normalized NumPy array
              - dense_projection: scalar projection of BGE-Small 384-d embedding
        """
        combined: Dict[str, Any] = {}
        if features and isinstance(features, dict):
            combined.update(features)
        combined.update(kwargs)

        tech = combined.get("tech") if isinstance(combined.get("tech"), dict) else {}
        bert = combined.get("bert") if isinstance(combined.get("bert"), dict) else {}
        articles = combined.get("articles") or []

        # Resolve FinBERT analyzer if text/headline provided without sentiment
        txt = combined.get("text") or combined.get("headline")
        if (not bert) and (combined.get("net_sentiment") is None and combined.get("net_score") is None) and txt:
            analyzer = self.finbert_analyzer or kwargs.get("finbert_analyzer")
            if analyzer is None and FinBERTAnalyzer is not None:
                analyzer = FinBERTAnalyzer.get_instance()
            if analyzer is not None:
                try:
                    bert = analyzer.analyze(str(txt))
                except Exception as e:
                    logger.debug(f"FinBERT analysis failed: {e}")

        # 1. Net Sentiment & Probabilities
        net_score = combined.get("net_sentiment")
        if net_score is None:
            net_score = combined.get("net_score")
        if net_score is None:
            net_score = bert.get("net_score", bert.get("net_sentiment", 0.0))
        net_sentiment = _safe_float(net_score, 0.0)
        net_sentiment = float(np.clip(net_sentiment, -1.0, 1.0))

        pos_val = combined.get("positive_prob", bert.get("positive", bert.get("positive_prob", 0.0)))
        pos_prob = _safe_float(pos_val, 0.0)
        if pos_prob <= 0.0:
            pos_prob = float(np.clip((net_sentiment + 1.0) / 2.0, 0.0, 1.0))

        neg_val = combined.get("negative_prob", bert.get("negative", bert.get("negative_prob", 0.0)))
        neg_prob = _safe_float(neg_val, 0.0)
        if neg_prob <= 0.0:
            neg_prob = float(np.clip((1.0 - net_sentiment) / 2.0, 0.0, 1.0))

        neu_val = combined.get("neutral_prob", bert.get("neutral", bert.get("neutral_prob", 0.0)))
        neu_prob = _safe_float(neu_val, 0.0)
        if neu_prob <= 0.0:
            neu_prob = float(np.clip(1.0 - (pos_prob + neg_prob), 0.0, 1.0))

        # Apply calibration if calibrator is provided and probabilities were not pre-calibrated
        cal = self.calibrator or kwargs.get("calibrator")
        if cal is not None and hasattr(cal, "calibrate_probs") and not bert.get("calibrated", False):
            try:
                cal_res = cal.calibrate_probs({"positive": pos_prob, "negative": neg_prob, "neutral": neu_prob})
                pos_prob = float(cal_res.get("positive", pos_prob))
                neg_prob = float(cal_res.get("negative", neg_prob))
                neu_prob = float(cal_res.get("neutral", neu_prob))
                net_sentiment = float(cal_res.get("net_score", net_sentiment))
            except Exception as e:
                logger.debug(f"Calibration application failed: {e}")

        # 2. FinBERT Logits
        logits = combined.get("sentiment_logits") or bert.get("logits")
        if logits and isinstance(logits, (list, tuple, np.ndarray)) and len(logits) >= 3:
            logit_pos = _safe_float(logits[0], 0.0)
            logit_neg = _safe_float(logits[1], 0.0)
            logit_neu = _safe_float(logits[2], 0.0)
        else:
            # Recover pseudo-logits via log-probabilities
            logit_pos = float(np.log(np.clip(pos_prob if pos_prob > 0 else 0.33, 1e-4, 1.0)))
            logit_neg = float(np.log(np.clip(neg_prob if neg_prob > 0 else 0.33, 1e-4, 1.0)))
            logit_neu = float(np.log(np.clip(neu_prob if neu_prob > 0 else 0.34, 1e-4, 1.0)))

        # 3. Technical Indicators
        rsi_val = combined.get("rsi_14")
        if rsi_val is None:
            rsi_val = combined.get("rsi", tech.get("rsi_14", tech.get("rsi", 50.0)))
        rsi_14 = float(np.clip(_safe_float(rsi_val, 50.0), 0.0, 100.0))

        mom_val = combined.get("momentum_5d")
        if mom_val is None:
            mom_val = tech.get("momentum_5d", tech.get("mom_5d", 0.0))
        momentum_5d = float(np.clip(_safe_float(mom_val, 0.0), -25.0, 25.0))

        vol_val = combined.get("volume_z_score")
        if vol_val is None:
            vol_val = tech.get("volume_z_score", tech.get("vol_z", 0.0))
        volume_z_score = float(np.clip(_safe_float(vol_val, 0.0), -3.0, 3.0))

        atr_val = combined.get("atr_volatility")
        if atr_val is None:
            atr_val = tech.get("atr_volatility", tech.get("atr_14", 0.0))
        atr_volatility = float(np.clip(_safe_float(atr_val, 0.0), 0.0, 20.0))

        above_sma = bool(combined.get("above_sma20", tech.get("above_sma20", False)))
        above_sma_val = 1.0 if above_sma else 0.0

        # 4. Catalyst Clarity & Text Features
        clarity_val = combined.get("catalyst_clarity", combined.get("c_clarity"))
        if clarity_val is None:
            # Compute clarity score if not explicitly passed
            c_score = 0.10
            has_arts = bool(articles or bert.get("headline"))
            if has_arts:
                c_score = 0.40 + min(0.25, abs(net_sentiment) * 0.25)
                # Check for standard catalyst keywords
                catalyst_keywords = [
                    "earnings", "beat", "surge", "breakout", "deal", "acquisition",
                    "merger", "contract", "revenue", "profit", "fda", "approval",
                    "guidance", "raise", "dividend", "buyback", "expansion", "patent"
                ]
                all_text = " ".join(art.get("title", "") for art in articles).lower()
                if bert.get("headline"):
                    all_text = f"{all_text} {str(bert.get('headline')).lower()}"
                if any(kw in all_text for kw in catalyst_keywords):
                    c_score += 0.30
            catalyst_clarity = float(np.clip(c_score, 0.05, 1.0))
        else:
            catalyst_clarity = float(np.clip(_safe_float(clarity_val, 0.10), 0.05, 1.0))

        # Dense text embedding (384 dims from FastEmbed BGE-Small)
        dense_emb = combined.get("dense_text_embedding", combined.get("text_embedding"))
        text_anchor_sim = 0.0
        dense_proj = 0.0

        if dense_emb is not None:
            try:
                emb_arr = np.asarray(dense_emb, dtype=np.float64).flatten()
                if emb_arr.shape[0] == 384:
                    norm = np.linalg.norm(emb_arr)
                    if norm > 1e-12:
                        norm_emb = emb_arr / norm
                        text_anchor_sim = float(np.dot(norm_emb, CATALYST_ANCHOR_VECTOR))
                        dense_proj = text_anchor_sim
            except Exception as e:
                logger.debug(f"Could not compute dense embedding similarity: {e}")

        # Explicit text anchor similarity override
        explicit_sim = combined.get("text_anchor_similarity")
        if explicit_sim is not None:
            text_anchor_sim = float(np.clip(_safe_float(explicit_sim, 0.0), -1.0, 1.0))

        features_dict = {
            "net_sentiment": net_sentiment,
            "positive_prob": pos_prob,
            "negative_prob": neg_prob,
            "neutral_prob": neu_prob,
            "sentiment_logit_pos": logit_pos,
            "sentiment_logit_neg": logit_neg,
            "sentiment_logit_neu": logit_neu,
            "rsi_14": rsi_14,
            "momentum_5d": momentum_5d,
            "volume_z_score": volume_z_score,
            "atr_volatility": atr_volatility,
            "above_sma20": above_sma_val,
            "catalyst_clarity": catalyst_clarity,
            "text_anchor_similarity": text_anchor_sim,
        }

        feature_vector = np.array([
            net_sentiment,
            pos_prob,
            neg_prob,
            neu_prob,
            logit_pos,
            logit_neg,
            logit_neu,
            rsi_14,
            momentum_5d,
            volume_z_score,
            atr_volatility,
            above_sma_val,
            catalyst_clarity,
            text_anchor_sim,
        ], dtype=np.float64)

        return features_dict, feature_vector, dense_proj

    def predict_edge_probability(
        self,
        features: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> float:
        """
        Predicts continuous trade edge probability in [0.0, 1.0].

        Executes the highest available tier:
          Tier 1 (LightGBM) -> Tier 2 (Scikit-Learn) -> Tier 3 (NumPy Fallback).
        """
        features_dict, feature_vector, dense_proj = self.extract_features(features, **kwargs)

        # Tier 1: LightGBM
        if self.tier_override == 1 or (self.tier_override is None and self.lgbm_model and self.lgbm_model.is_available()):
            try:
                if self.lgbm_model and self.lgbm_model.is_available():
                    prob = self.lgbm_model.predict_proba(feature_vector)
                    return float(np.clip(prob, 0.0, 1.0))
            except Exception as e:
                logger.warning(f"Tier 1 LightGBM execution failed, falling back: {e}")

        # Tier 2: Scikit-Learn
        if self.tier_override == 2 or (self.tier_override is None and self.sklearn_model and self.sklearn_model.is_available()):
            try:
                if self.sklearn_model and self.sklearn_model.is_available():
                    prob = self.sklearn_model.predict_proba(feature_vector)
                    return float(np.clip(prob, 0.0, 1.0))
            except Exception as e:
                logger.warning(f"Tier 2 Sklearn execution failed, falling back: {e}")

        # Tier 3: Pure NumPy Fallback
        prob = self.numpy_fallback.predict_proba(features_dict, dense_projection=dense_proj)
        return float(np.clip(prob, 0.0, 1.0))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Batch prediction interface: accepts (N, 14) array and returns (N, 2) probabilities.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        N = X.shape[0]
        probs = np.zeros(N, dtype=np.float64)
        for i in range(N):
            row = X[i]
            features_dict = {name: float(row[j]) for j, name in enumerate(FEATURE_NAMES) if j < len(row)}
            probs[i] = self.predict_edge_probability(features=features_dict)

        return np.column_stack([1.0 - probs, probs])

    def incorporate_trade_feedback(
        self,
        feedback_data: Optional[Union[List[Dict[str, Any]], Any, str, Path]] = None,
        learning_rate: float = 0.05,
        loss_penalty_multiplier: float = 1.5,
        epochs: int = 5,
    ) -> Dict[str, Any]:
        """
        Incorporates trade feedback (experience replay / mistake learning) into the fusion models.
        Penalizes feature combinations that resulted in stopped-out trades (y=0) while reinforcing
        winning setups (y=1). Preserves mathematical monotonicity across all tiers.
        """
        records: List[Dict[str, Any]] = []

        if feedback_data is None:
            if FEEDBACK_PARQUET_FILE.exists() and HAS_PANDAS:
                try:
                    df = pd.read_parquet(FEEDBACK_PARQUET_FILE)
                    records = df.to_dict(orient="records")
                except Exception as e:
                    logger.warning(f"Failed to read parquet feedback file {FEEDBACK_PARQUET_FILE}: {e}")
            if not records and FEEDBACK_JSON_FILE.exists():
                try:
                    with open(FEEDBACK_JSON_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            records = data
                except Exception as e:
                    logger.warning(f"Failed to read json feedback file {FEEDBACK_JSON_FILE}: {e}")
        elif isinstance(feedback_data, (str, Path)):
            p = Path(feedback_data)
            if p.suffix == ".parquet" and HAS_PANDAS:
                df = pd.read_parquet(p)
                records = df.to_dict(orient="records")
            elif p.suffix == ".json" or p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    records = json.load(f)
        elif hasattr(feedback_data, "to_dict"):
            records = feedback_data.to_dict(orient="records")
        elif isinstance(feedback_data, list):
            records = list(feedback_data)

        if not records:
            return {
                "status": "no_data",
                "samples_processed": 0,
                "message": "No trade feedback data found to incorporate"
            }

        feedback_samples: List[Tuple[Dict[str, Any], int]] = []
        for r in records:
            label_val = r.get("label")
            if label_val is None:
                pnl_val = float(r.get("pnl", r.get("realized_pnl", 0.0)))
                label_val = 1 if pnl_val > 0 else 0
            feedback_samples.append((r, int(label_val)))

        # Update pure-NumPy fallback (Tier 3)
        np_res = self.numpy_fallback.update_from_feedback(
            feedback_samples=feedback_samples,
            learning_rate=learning_rate,
            loss_penalty_multiplier=loss_penalty_multiplier,
            epochs=epochs,
        )

        wins = sum(1 for _, l in feedback_samples if l == 1)
        losses = sum(1 for _, l in feedback_samples if l == 0)

        # Update Tier 2 if available
        sklearn_updated = False
        if HAS_SKLEARN and self.sklearn_model and self.sklearn_model.is_available() and len(feedback_samples) >= 4:
            try:
                X_list = []
                y_list = []
                for feat, lbl in feedback_samples:
                    _, fvec, _ = self.extract_features(feat)
                    X_list.append(fvec)
                    y_list.append(lbl)
                X_arr = np.array(X_list)
                y_arr = np.array(y_list)
                if len(np.unique(y_arr)) > 1:
                    sklearn_updated = self.sklearn_model.update_with_feedback(
                        X_arr, y_arr, loss_penalty_multiplier=loss_penalty_multiplier
                    )
            except Exception as e:
                logger.warning(f"Sklearn model update on feedback skipped: {e}")

        return {
            "status": "success",
            "samples_processed": len(feedback_samples),
            "wins": wins,
            "losses": losses,
            "active_tier": self.active_tier,
            "loss_penalty_applied": True,
            "loss_penalty_multiplier": loss_penalty_multiplier,
            "numpy_fallback_update": np_res,
            "sklearn_updated": sklearn_updated,
        }

    def train_on_feedback(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for incorporate_trade_feedback."""
        return self.incorporate_trade_feedback(*args, **kwargs)

    def retrain(
        self,
        X: Union[np.ndarray, List[Dict[str, Any]], List[Tuple[Dict[str, Any], int]], Any],
        y: Optional[Union[np.ndarray, List[int]]] = None,
        learning_rate: float = 0.05,
        epochs: int = 5,
    ) -> Dict[str, Any]:
        """
        Retrains the fusion engine on a dataset of feature dictionaries, tuples, or arrays.
        """
        samples: List[Tuple[Dict[str, Any], int]] = []
        if isinstance(X, list) and (y is None or len(y) == 0):
            for item in X:
                if isinstance(item, tuple) and len(item) == 2:
                    samples.append((item[0], int(item[1])))
                elif isinstance(item, dict):
                    lbl = int(item.get("label", 1 if float(item.get("pnl", 0.0)) > 0 else 0))
                    samples.append((item, lbl))
        elif y is not None:
            if isinstance(X, np.ndarray):
                for i in range(len(X)):
                    row = X[i]
                    fdict = {name: float(row[j]) for j, name in enumerate(FEATURE_NAMES) if j < len(row)}
                    samples.append((fdict, int(y[i])))
            elif isinstance(X, list):
                for i in range(len(X)):
                    samples.append((X[i], int(y[i])))

        if not samples:
            return {"status": "error", "message": "Invalid training input or empty samples"}

        np_res = self.numpy_fallback.update_from_feedback(samples, learning_rate=learning_rate, epochs=epochs)

        # Also update Sklearn if available
        sklearn_updated = False
        if HAS_SKLEARN and self.sklearn_model and self.sklearn_model.is_available() and len(samples) >= 4:
            try:
                X_list = []
                y_list = []
                for feat, lbl in samples:
                    _, fvec, _ = self.extract_features(feat)
                    X_list.append(fvec)
                    y_list.append(lbl)
                X_arr = np.array(X_list)
                y_arr = np.array(y_list)
                if len(np.unique(y_arr)) > 1:
                    sklearn_updated = self.sklearn_model.update_with_feedback(X_arr, y_arr)
            except Exception:
                pass

        return {
            "status": "success",
            "samples": len(samples),
            "numpy_update": np_res,
            "sklearn_updated": sklearn_updated,
            "monotonic_weight_sent": self.numpy_fallback.w_sent,
        }


# ============================================================================
# SINGLETON CACHE
# ============================================================================

_ENGINE_LOCK = threading.Lock()
_ENGINE_INSTANCE: Optional[MultiModalFusionEngine] = None


def get_fusion_engine(
    tier: Optional[int] = None,
    finbert_analyzer: Optional[Any] = None,
    calibrator: Optional[Any] = None,
) -> MultiModalFusionEngine:
    """
    Returns the singleton MultiModalFusionEngine instance.
    Thread-safe cached factory with dependency injection support.
    """
    global _ENGINE_INSTANCE
    with _ENGINE_LOCK:
        if _ENGINE_INSTANCE is None or (tier is not None and _ENGINE_INSTANCE.tier_override != tier):
            _ENGINE_INSTANCE = MultiModalFusionEngine(
                tier=tier,
                finbert_analyzer=finbert_analyzer,
                calibrator=calibrator,
            )
        else:
            if finbert_analyzer is not None:
                _ENGINE_INSTANCE.finbert_analyzer = finbert_analyzer
            if calibrator is not None:
                _ENGINE_INSTANCE.calibrator = calibrator
        return _ENGINE_INSTANCE


def reset_fusion_engine() -> None:
    """Resets the singleton fusion engine instance (useful for unit testing)."""
    global _ENGINE_INSTANCE
    with _ENGINE_LOCK:
        _ENGINE_INSTANCE = None
