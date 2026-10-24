"""
Automated Weekly Rolling Retraining Engine (Continuous Learning Upgrade A).

Features:
1. Slices market history to a rolling chronological window (default: trailing 90 trading days).
2. Supports completely offline operation using cached OHLCV bars in `data/historical_bars/`.
3. Re-calibrates bot parameters across all 5 bot archetypes using WalkForwardCalibrator.
4. Enforces the strict Overfitting Detection Gate (`check_overfitting_gate`):
   - Degraded / overfitted strategy candidates (OOS return <= 0, OOS PF < 1.0, severe decay)
     are strictly REJECTED.
   - Previous optimal parameters and weights are safely preserved.
5. Retrains the MultiModalFusionEngine on rolling-window market features and incorporates
   accumulated trade feedback from experience replay.
6. Updates `data/bot_calibrated_parameters.json` and produces audit reports.
"""

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from src.backtest_engine import (
    HistoricalBacktestEngine,
    HistoricalDataFetcher,
    calculate_performance_metrics,
)
from src.calibration import (
    DEFAULT_PERSONA_GRIDS,
    WalkForwardCalibrator,
    check_overfitting_gate,
    calculate_fitness_score,
)
from src.fusion_engine import get_fusion_engine, MultiModalFusionEngine
from src.paper_trader import (
    AggressiveBreakoutBot,
    BotArchetype,
    ConservativeSwingBot,
    ContrarianDipBuyerBot,
    FinBERTCatalystHunterBot,
    TrendFollowerBot,
    TradeFeedbackStore,
)

logger = logging.getLogger("RollingRetrainer")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CALIBRATED_PARAMS_FILE = DEFAULT_DATA_DIR / "bot_calibrated_parameters.json"
DEFAULT_UNIVERSE = ["^NSEI", "NIFTYBEES.NS", "NVDA", "AAPL", "MSFT", "SPY"]


class RollingRetrainingEngine:
    """
    Automated Rolling Retraining Engine.
    Executes walk-forward re-calibration on a rolling window of recent trading bars
    while enforcing the strict overfitting gate to safeguard production bot configurations.
    """

    def __init__(
        self,
        rolling_window_days: int = 90,
        in_sample_pct: float = 0.70,
        universe: Optional[List[str]] = None,
        offline: bool = True,
        data_dir: Optional[Union[str, Path]] = None,
        params_file: Optional[Union[str, Path]] = None,
        fusion_engine: Optional[MultiModalFusionEngine] = None,
    ):
        self.rolling_window_days = int(rolling_window_days)
        self.in_sample_pct = float(in_sample_pct)
        self.universe = list(universe or DEFAULT_UNIVERSE)
        self.offline = bool(offline)
        self.data_dir = Path(data_dir).resolve() if data_dir else DEFAULT_DATA_DIR
        self.params_file = Path(params_file).resolve() if params_file else DEFAULT_CALIBRATED_PARAMS_FILE
        self.fusion_engine = fusion_engine or get_fusion_engine()
        self.feedback_store = TradeFeedbackStore(data_dir=self.data_dir)

    def load_existing_parameters(self) -> Dict[str, Any]:
        """Loads currently active / optimal parameters from JSON storage."""
        if self.params_file.exists():
            try:
                with open(self.params_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing parameters from {self.params_file}: {e}")
        return {}

    def fetch_rolling_bars(self, full_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetches historical daily bars for universe and trims each symbol to trailing rolling window days.
        Guarantees offline operation when requested.
        """
        if full_data is None:
            bars_cache_dir = self.data_dir / "historical_bars" if (self.data_dir / "historical_bars").exists() else self.data_dir
            fetcher = HistoricalDataFetcher(cache_dir=bars_cache_dir, offline=self.offline)
            # 1 year covers at least 250 trading bars, enough for a 90-day rolling window
            full_data = fetcher.fetch_universe_bars(symbols=self.universe, years=1)

        rolling_data: Dict[str, pd.DataFrame] = {}
        for sym, df in full_data.items():
            if df is not None and not df.empty:
                # Take trailing rolling_window_days
                sliced = df.tail(self.rolling_window_days).copy()
                if len(sliced) >= 10:  # Minimal threshold to evaluate
                    rolling_data[sym] = sliced
                else:
                    rolling_data[sym] = df.copy()

        return rolling_data

    def recalibrate_bots(
        self,
        rolling_data: Dict[str, pd.DataFrame],
        custom_grids: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """
        Recalibrates all 5 bot personas over rolling window data and strictly evaluates the overfitting gate.
        Rejects degraded candidates and preserves previous optimal parameters.
        """
        existing_params = self.load_existing_parameters()
        calibrator = WalkForwardCalibrator(in_sample_pct=self.in_sample_pct)

        if not rolling_data or all(df is None or df.empty for df in rolling_data.values()):
            logger.warning("[RollingRetrainer] No valid rolling bar data provided. Preserving existing parameters.")
            return {
                "bot_evaluations": {},
                "updated_params": existing_params,
                "accepted_count": 0,
                "rejected_count": 0,
            }

        persona_classes: List[Type[BotArchetype]] = [
            ConservativeSwingBot,
            AggressiveBreakoutBot,
            ContrarianDipBuyerBot,
            FinBERTCatalystHunterBot,
            TrendFollowerBot,
        ]

        bot_results: Dict[str, Any] = {}
        updated_params: Dict[str, Any] = dict(existing_params)

        for bot_cls in persona_classes:
            sample_b = bot_cls()
            bot_id = sample_b.bot_id
            grid = (custom_grids or {}).get(bot_id, DEFAULT_PERSONA_GRIDS.get(bot_id))

            prev_entry = existing_params.get(bot_id, {})
            prev_status = prev_entry.get("status", "unknown")
            prev_params = prev_entry.get("parameters", prev_entry.get("best_params", sample_b.get_params()))

            try:
                # Run walk-forward calibration on rolling slice
                cal_res = calibrator.calibrate(
                    bot_cls,
                    rolling_data,
                    param_grid=grid,
                    in_sample_pct=self.in_sample_pct
                )

                is_metrics = cal_res.get("in_sample_metrics", {})
                oos_metrics = cal_res.get("out_of_sample_metrics", {})

                # Enforce the strict overfitting detection gate
                is_overfit, gate_reason = check_overfitting_gate(is_metrics, oos_metrics)
            except Exception as cal_err:
                logger.error(f"[RollingRetrainer] Error calibrating '{bot_id}': {cal_err}. Rejecting candidate.")
                is_overfit = True
                gate_reason = f"Calibration error: {cal_err}"
                cal_res = {"best_params": prev_params, "in_sample_metrics": {}, "out_of_sample_metrics": {}}
                is_metrics = {}
                oos_metrics = {}

            if is_overfit:
                # STRICT REJECTION: Degraded strategy candidate
                logger.warning(
                    f"[RollingRetrainer] REJECTED candidate for bot '{bot_id}' due to overfitting: {gate_reason}. "
                    f"Safely retaining previous optimal weights."
                )
                bot_results[bot_id] = {
                    "bot_id": bot_id,
                    "action": "rejected",
                    "reason": gate_reason,
                    "is_overfit": True,
                    "candidate_params": cal_res.get("best_params", {}),
                    "candidate_is_metrics": is_metrics,
                    "candidate_oos_metrics": oos_metrics,
                    "retained_params": prev_params,
                    "status": "degraded" if prev_status == "degraded" else "retained_optimal",
                }
                # Keep existing parameters in updated_params
                if bot_id not in updated_params:
                    updated_params[bot_id] = {
                        "bot_id": bot_id,
                        "status": "retained_optimal",
                        "parameters": prev_params,
                        "best_params": prev_params,
                    }
                updated_params[bot_id]["last_rolling_evaluation"] = {
                    "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "candidate_rejected": True,
                    "rejection_reason": gate_reason,
                }
            else:
                # ACCEPTANCE: Strategy candidate passed all out-of-sample health criteria
                logger.info(f"[RollingRetrainer] ACCEPTED candidate for bot '{bot_id}'. Updating bot parameters.")
                bot_results[bot_id] = {
                    "bot_id": bot_id,
                    "action": "accepted",
                    "reason": "Passed overfitting gate with positive OOS edge",
                    "is_overfit": False,
                    "new_params": cal_res.get("best_params", {}),
                    "is_metrics": is_metrics,
                    "oos_metrics": oos_metrics,
                    "status": "optimal",
                }

                # Update in output dictionary
                updated_params[bot_id] = {
                    "bot_id": bot_id,
                    "calibrated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "parameters": cal_res.get("best_params", {}),
                    "best_params": cal_res.get("best_params", {}),
                    "status": "optimal",
                    "in_sample_metrics": is_metrics,
                    "out_of_sample_metrics": oos_metrics,
                    "full_period_metrics": cal_res.get("full_period_metrics", {}),
                    "walk_forward_score": cal_res.get("walk_forward_score", 0.0),
                    "last_rolling_retrain": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }

        return {
            "bot_evaluations": bot_results,
            "updated_params": updated_params,
            "accepted_count": sum(1 for r in bot_results.values() if r["action"] == "accepted"),
            "rejected_count": sum(1 for r in bot_results.values() if r["action"] == "rejected"),
        }

    def retrain_fusion_engine(
        self,
        rolling_data: Dict[str, pd.DataFrame],
        learning_rate: float = 0.05,
        epochs: int = 5,
    ) -> Dict[str, Any]:
        """
        Retrains the MultiModalFusionEngine on rolling bar feature dynamics
        and incorporates accumulated trade feedback.
        """
        # 1. First incorporate trade feedback (experience replay / mistake learning)
        feedback_report = self.fusion_engine.incorporate_trade_feedback(
            learning_rate=learning_rate,
            epochs=epochs
        )

        # 2. Extract feature-label samples from rolling price bars
        samples: List[Tuple[Dict[str, Any], int]] = []
        for sym, df in rolling_data.items():
            if df is None or len(df) < 20:
                continue
            closes = df["close"].values
            vols = df.get("volume", pd.Series(np.ones(len(df)) * 100000)).values
            # Compute rolling features and forward 3-day return label
            for i in range(15, len(closes) - 3):
                cur_p = float(closes[i])
                fwd_p = float(closes[i + 3])
                label = 1 if fwd_p > cur_p else 0

                p_5d_ago = float(closes[i - 5]) if i >= 5 else cur_p
                mom5d = ((cur_p - p_5d_ago) / p_5d_ago * 100.0) if p_5d_ago > 0 else 0.0

                lookback_sma = closes[max(0, i - 19) : i + 1]
                sma20 = float(np.mean(lookback_sma)) if len(lookback_sma) > 0 else cur_p

                fdict = {
                    "net_sentiment": 0.20 if mom5d > 1.0 else (-0.20 if mom5d < -1.0 else 0.0),
                    "positive_prob": 0.60 if mom5d > 1.0 else 0.20,
                    "negative_prob": 0.20 if mom5d > 1.0 else 0.60,
                    "neutral_prob": 0.20,
                    "rsi_14": 55.0 if mom5d > 0 else 45.0,
                    "momentum_5d": float(np.clip(mom5d, -20.0, 20.0)),
                    "volume_z_score": 0.5 if vols[i] > np.mean(vols[: i + 1]) else -0.5,
                    "atr_volatility": 1.5,
                    "above_sma20": 1.0 if cur_p >= sma20 else 0.0,
                    "catalyst_clarity": 0.40,
                    "text_anchor_similarity": 0.10,
                }
                samples.append((fdict, label))

        # Retrain on extracted samples if available via MultiModalFusionEngine interface
        retrain_res = {}
        if samples:
            retrain_res = self.fusion_engine.retrain(
                samples[:150],  # Bound batch size
                learning_rate=learning_rate,
                epochs=epochs,
            )

        return {
            "status": "success",
            "samples_trained": len(samples),
            "trade_feedback_result": feedback_report,
            "fusion_retrain_result": retrain_res,
            "active_tier": self.fusion_engine.active_tier,
            "monotonic_weight_sent": self.fusion_engine.numpy_fallback.w_sent,
        }

    def run_retraining(
        self,
        full_data: Optional[Dict[str, pd.DataFrame]] = None,
        custom_grids: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        save_parameters: bool = True,
    ) -> Dict[str, Any]:
        """
        Coordinates the complete weekly rolling retraining workflow:
        1. Ingests / slices rolling window bars.
        2. Recalibrates bots while strictly enforcing the overfitting gate.
        3. Retrains MultiModalFusionEngine with experience replay.
        4. Persists safe parameter updates.
        """
        start_time = datetime.datetime.now(datetime.timezone.utc)
        logger.info(
            f"[RollingRetrainer] Starting rolling retraining over trailing "
            f"{self.rolling_window_days} trading days (Offline={self.offline})."
        )

        # 1. Fetch rolling bars
        rolling_data = self.fetch_rolling_bars(full_data=full_data)

        # 2. Recalibrate bots with overfitting gate
        bot_recal_res = self.recalibrate_bots(rolling_data, custom_grids=custom_grids)

        # 3. Retrain fusion engine
        fusion_res = self.retrain_fusion_engine(rolling_data)

        # 4. Save parameters if requested
        if save_parameters and bot_recal_res.get("updated_params"):
            try:
                self.params_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.params_file, "w", encoding="utf-8") as f:
                    json.dump(bot_recal_res["updated_params"], f, indent=2, default=str)
                logger.info(f"[RollingRetrainer] Persisted safe parameters to {self.params_file}")
            except Exception as e:
                logger.error(f"[RollingRetrainer] Failed to save updated parameters: {e}")

        elapsed_sec = round((datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds(), 2)

        return {
            "status": "success",
            "retrained_at": start_time.isoformat(),
            "elapsed_seconds": elapsed_sec,
            "rolling_window_days": self.rolling_window_days,
            "universe": list(rolling_data.keys()),
            "offline": self.offline,
            "overfitting_gate_enforced": True,
            "accepted_count": bot_recal_res["accepted_count"],
            "rejected_count": bot_recal_res["rejected_count"],
            "bot_evaluations": bot_recal_res["bot_evaluations"],
            "fusion_retraining": fusion_res,
        }


def run_rolling_retraining(
    window_days: int = 90,
    in_sample_pct: float = 0.70,
    offline: bool = True,
    universe: Optional[List[str]] = None,
    save_parameters: bool = True,
) -> Dict[str, Any]:
    """
    Convenience functional interface for automated rolling retraining.
    """
    engine = RollingRetrainingEngine(
        rolling_window_days=window_days,
        in_sample_pct=in_sample_pct,
        universe=universe,
        offline=offline,
    )
    return engine.run_retraining(save_parameters=save_parameters)
