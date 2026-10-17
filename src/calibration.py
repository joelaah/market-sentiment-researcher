"""
Walk-Forward Parameter Calibration & Optimization Engine (Milestone 2).
Provides:
1. Strict chronological lookahead-free data partitioning (67% In-Sample / 33% Out-of-Sample).
2. Composite regularized fitness function balancing Profit Factor, Win Rate, Net Return,
   and Max Drawdown with minimum trade count penalties.
3. Multi-portfolio simultaneous batch replay for high-performance sub-second evaluation.
4. Curated parameter search spaces per persona (8 candidate sets each).
5. WalkForwardCalibrator and calibrate_all_personas producing standardized Supabase/JSON schemas.
"""

import datetime
import itertools
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from src.backtest_engine import (
    HistoricalBacktestEngine,
    HistoricalDataFetcher,
    calculate_performance_metrics,
)
from src.paper_trader import (
    AggressiveBreakoutBot,
    BotArchetype,
    ConservativeSwingBot,
    ContrarianDipBuyerBot,
    FinBERTCatalystHunterBot,
    TrendFollowerBot,
)

logger = logging.getLogger("CalibrationEngine")


# ============================================================================
# CURATED PARAMETER SEARCH SPACES PER PERSONA (8 CANDIDATES EACH)
# ============================================================================

CONSERVATIVE_SWING_GRID: List[Dict[str, Any]] = [
    {"stop_loss_pct": -5.0, "take_profit_pct": 10.0, "min_momentum_5d": 0.5, "max_positions": 4},
    {"stop_loss_pct": -5.0, "take_profit_pct": 12.0, "min_momentum_5d": 0.0, "max_positions": 5},
    {"stop_loss_pct": -7.5, "take_profit_pct": 10.0, "min_momentum_5d": 0.0, "max_positions": 5},
    {"stop_loss_pct": -7.5, "take_profit_pct": 12.0, "min_momentum_5d": 0.0, "max_positions": 5},  # Baseline
    {"stop_loss_pct": -7.5, "take_profit_pct": 15.0, "min_momentum_5d": 0.5, "max_positions": 4},
    {"stop_loss_pct": -10.0, "take_profit_pct": 12.0, "min_momentum_5d": 0.0, "max_positions": 5},
    {"stop_loss_pct": -10.0, "take_profit_pct": 15.0, "min_momentum_5d": 0.5, "max_positions": 5},
    {"stop_loss_pct": -6.0, "take_profit_pct": 14.0, "min_momentum_5d": 0.5, "max_positions": 4},
]

AGGRESSIVE_BREAKOUT_GRID: List[Dict[str, Any]] = [
    {"stop_loss_pct": -4.0, "take_profit_pct": 14.0, "min_momentum_5d": 1.5, "allocation_pct": 0.20},
    {"stop_loss_pct": -4.0, "take_profit_pct": 18.0, "min_momentum_5d": 2.0, "allocation_pct": 0.25},
    {"stop_loss_pct": -5.0, "take_profit_pct": 14.0, "min_momentum_5d": 2.0, "allocation_pct": 0.20},
    {"stop_loss_pct": -5.0, "take_profit_pct": 18.0, "min_momentum_5d": 2.0, "allocation_pct": 0.25},  # Baseline
    {"stop_loss_pct": -5.0, "take_profit_pct": 22.0, "min_momentum_5d": 2.5, "allocation_pct": 0.25},
    {"stop_loss_pct": -6.5, "take_profit_pct": 18.0, "min_momentum_5d": 2.0, "allocation_pct": 0.20},
    {"stop_loss_pct": -6.5, "take_profit_pct": 22.0, "min_momentum_5d": 2.5, "allocation_pct": 0.25},
    {"stop_loss_pct": -4.5, "take_profit_pct": 16.0, "min_momentum_5d": 1.8, "allocation_pct": 0.25},
]

CONTRARIAN_DIP_BUYER_GRID: List[Dict[str, Any]] = [
    {"stop_loss_pct": -4.5, "take_profit_pct": 6.0, "dip_threshold_pct": -1.0},
    {"stop_loss_pct": -4.5, "take_profit_pct": 8.0, "dip_threshold_pct": -1.5},
    {"stop_loss_pct": -6.0, "take_profit_pct": 6.0, "dip_threshold_pct": -1.5},
    {"stop_loss_pct": -6.0, "take_profit_pct": 8.0, "dip_threshold_pct": -1.5},  # Baseline
    {"stop_loss_pct": -6.0, "take_profit_pct": 10.0, "dip_threshold_pct": -2.0},
    {"stop_loss_pct": -8.0, "take_profit_pct": 8.0, "dip_threshold_pct": -2.0},
    {"stop_loss_pct": -8.0, "take_profit_pct": 10.0, "dip_threshold_pct": -2.5},
    {"stop_loss_pct": -5.0, "take_profit_pct": 7.0, "dip_threshold_pct": -1.5},
]

FINBERT_CATALYST_HUNTER_GRID: List[Dict[str, Any]] = [
    {"stop_loss_pct": -5.0, "take_profit_pct": 8.0, "conviction_threshold": 0.35},
    {"stop_loss_pct": -5.0, "take_profit_pct": 10.0, "conviction_threshold": 0.50},
    {"stop_loss_pct": -7.0, "take_profit_pct": 8.0, "conviction_threshold": 0.50},
    {"stop_loss_pct": -7.0, "take_profit_pct": 10.0, "conviction_threshold": 0.50},  # Baseline
    {"stop_loss_pct": -7.0, "take_profit_pct": 14.0, "conviction_threshold": 0.65},
    {"stop_loss_pct": -9.0, "take_profit_pct": 10.0, "conviction_threshold": 0.50},
    {"stop_loss_pct": -9.0, "take_profit_pct": 14.0, "conviction_threshold": 0.65},
    {"stop_loss_pct": -6.0, "take_profit_pct": 12.0, "conviction_threshold": 0.45},
]

TREND_FOLLOWER_GRID: List[Dict[str, Any]] = [
    {"stop_loss_pct": -6.0, "take_profit_pct": 12.0, "min_momentum_5d": 0.0},
    {"stop_loss_pct": -6.0, "take_profit_pct": 15.0, "min_momentum_5d": 0.5},
    {"stop_loss_pct": -8.0, "take_profit_pct": 12.0, "min_momentum_5d": -0.5},
    {"stop_loss_pct": -8.0, "take_profit_pct": 15.0, "min_momentum_5d": 0.0},  # Baseline
    {"stop_loss_pct": -8.0, "take_profit_pct": 20.0, "min_momentum_5d": 0.5},
    {"stop_loss_pct": -10.0, "take_profit_pct": 15.0, "min_momentum_5d": -0.5},
    {"stop_loss_pct": -10.0, "take_profit_pct": 20.0, "min_momentum_5d": 0.0},
    {"stop_loss_pct": -7.5, "take_profit_pct": 16.0, "min_momentum_5d": 0.0},
]

DEFAULT_PERSONA_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "conservative_swing": CONSERVATIVE_SWING_GRID,
    "aggressive_breakout": AGGRESSIVE_BREAKOUT_GRID,
    "contrarian_dip_buyer": CONTRARIAN_DIP_BUYER_GRID,
    "finbert_catalyst_hunter": FINBERT_CATALYST_HUNTER_GRID,
    "trend_follower": TREND_FOLLOWER_GRID,
}


# ============================================================================
# HELPER UTILITIES
# ============================================================================

def _to_json_safe(obj: Any) -> Any:
    """Recursively converts numpy, pandas, and datetime objects into JSON-safe standard Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    elif isinstance(obj, (np.floating, float)):
        return round(float(obj), 4) if math.isfinite(obj) else 0.0
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (pd.Timestamp, datetime.date, datetime.datetime)):
        return obj.isoformat()
    return obj


def generate_grid_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Expands a dictionary of parameter lists into a list of unique configuration dicts."""
    if not param_grid:
        return []
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = []
    for comb in itertools.product(*values):
        combinations.append(dict(zip(keys, comb)))
    return combinations


# ============================================================================
# CHRONOLOGICAL DATA SPLIT (ZERO LOOKAHEAD)
# ============================================================================

def split_chronological_data(
    data: Dict[str, pd.DataFrame],
    in_sample_pct: float = 0.67
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Partitions multi-asset historical bar data chronologically into:
    - in_sample_data (training/calibration window, default 67%)
    - out_of_sample_data (validation window, default 33%)

    Guarantees zero future information leakage:
    - All in-sample dates strictly precede out-of-sample dates.
    - Slices are strictly disjoint.
    - Preserves multi-asset calendar synchronization across holidays.
    """
    if not data:
        raise ValueError("No historical data provided for chronological split")

    all_dates = sorted(list(set.union(*[set(df.index) for df in data.values()])))
    total_bars = len(all_dates)

    # Minimum historical bars requirement
    if total_bars < 4 or (len(data) == 1 and total_bars < 20):
        raise ValueError(f"Insufficient historical bars for split (minimum 20 bars required for single asset, got {total_bars})")

    split_idx = int(round(total_bars * in_sample_pct))
    if split_idx <= 0 or split_idx >= total_bars:
        raise ValueError(f"Invalid in-sample partition: split index {split_idx} out of {total_bars} bars")

    split_date = all_dates[split_idx]

    train_data: Dict[str, pd.DataFrame] = {}
    val_data: Dict[str, pd.DataFrame] = {}

    for sym, df in data.items():
        train_slice = df[df.index < split_date].copy()
        val_slice = df[df.index >= split_date].copy()
        train_data[sym] = train_slice
        val_data[sym] = val_slice

    return train_data, val_data


# ============================================================================
# COMPOSITE REGULARIZED FITNESS FUNCTION
# ============================================================================

def calculate_fitness_score(
    metrics: Dict[str, float],
    min_trades: int = 3,
    weight_return: float = 1.0,
    weight_profit_factor: float = 0.8,
    weight_win_rate: float = 0.5,
    weight_drawdown: float = 1.5,
    max_pf_cap: float = 5.0
) -> float:
    """
    Computes a composite scalar fitness score from backtest performance metrics.

    Formula:
      If total_trades < min_trades -> -999.0 (immediate curve-fitting disqualification)
      Else -> w_ret * (NetRet% / 100) + w_pf * min(PF, max_pf_cap) + w_wr * (WR% / 100)
              - w_mdd * (MDD% / 100) + min(0.2, 0.01 * total_trades)
    """
    total_trades = int(metrics.get("total_trades", 0))
    if total_trades < min_trades:
        return -999.0

    net_return = float(metrics.get("net_return_pct", 0.0))
    profit_factor = min(float(metrics.get("profit_factor", 0.0)), max_pf_cap)
    win_rate = float(metrics.get("win_rate_pct", 0.0))
    max_drawdown = float(metrics.get("max_drawdown_pct", 0.0))

    trade_bonus = min(0.20, 0.01 * total_trades)

    score = (
        weight_return * (net_return / 100.0)
        + weight_profit_factor * profit_factor
        + weight_win_rate * (win_rate / 100.0)
        - weight_drawdown * (max_drawdown / 100.0)
        + trade_bonus
    )
    return round(float(score), 4)


class ObjectiveFunction:
    """Configurable objective function evaluator for backtest calibration."""

    def __init__(
        self,
        min_trades: int = 3,
        weight_return: float = 1.0,
        weight_profit_factor: float = 0.8,
        weight_win_rate: float = 0.5,
        weight_drawdown: float = 1.5,
        max_pf_cap: float = 5.0
    ):
        self.min_trades = min_trades
        self.weight_return = weight_return
        self.weight_profit_factor = weight_profit_factor
        self.weight_win_rate = weight_win_rate
        self.weight_drawdown = weight_drawdown
        self.max_pf_cap = max_pf_cap

    def evaluate(self, metrics: Dict[str, float]) -> float:
        """Calculates scalar fitness score from performance metrics dictionary."""
        return calculate_fitness_score(
            metrics=metrics,
            min_trades=self.min_trades,
            weight_return=self.weight_return,
            weight_profit_factor=self.weight_profit_factor,
            weight_win_rate=self.weight_win_rate,
            weight_drawdown=self.weight_drawdown,
            max_pf_cap=self.weight_drawdown
        )


# ============================================================================
# WALK-FORWARD CALIBRATOR ENGINE
# ============================================================================

class WalkForwardCalibrator:
    """
    Chronological parameter calibration and walk-forward validation engine.
    Uses multi-portfolio simultaneous batch replay for extreme speed.
    """

    def __init__(
        self,
        engine: Optional[HistoricalBacktestEngine] = None,
        objective: Optional[ObjectiveFunction] = None,
        in_sample_pct: float = 0.67,
        top_k_candidates: int = 3,
        min_trades: int = 3
    ):
        self.engine = engine or HistoricalBacktestEngine()
        self.objective = objective or ObjectiveFunction(min_trades=min_trades)
        self.in_sample_pct = in_sample_pct
        self.top_k_candidates = top_k_candidates
        self.min_trades = min_trades

    def split_data(
        self,
        data: Dict[str, pd.DataFrame],
        in_sample_pct: Optional[float] = None
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Splits multi-asset data into in-sample and out-of-sample slices chronologically."""
        return split_chronological_data(data, in_sample_pct=in_sample_pct or self.in_sample_pct)

    def compute_fitness(self, metrics: Dict[str, float]) -> float:
        """Calculates in-sample scalar fitness score."""
        return calculate_fitness_score(metrics, min_trades=self.min_trades)

    def compute_walk_forward_score(
        self,
        in_sample_metrics: Dict[str, float],
        out_of_sample_metrics: Dict[str, float]
    ) -> float:
        """
        Calculates composite walk-forward score balancing in-sample performance
        and out-of-sample robustness (0.6 * IS + 0.4 * OOS) with drawdown inflation penalties.
        """
        f_is = self.compute_fitness(in_sample_metrics)
        f_oos = calculate_fitness_score(out_of_sample_metrics, min_trades=1)

        # Severe in-sample failure
        if f_is <= -900.0:
            return -999.0

        # Drawdown inflation check (penalize if OOS drawdown balloons relative to IS)
        mdd_is = max(1.0, float(in_sample_metrics.get("max_drawdown_pct", 0.0)))
        mdd_oos = float(out_of_sample_metrics.get("max_drawdown_pct", 0.0))
        dd_ratio = mdd_oos / mdd_is
        dd_penalty = max(0.0, (dd_ratio - 2.0) * 0.5)

        score = 0.6 * f_is + 0.4 * f_oos - dd_penalty
        return round(float(score), 4)

    def calibrate(
        self,
        bot_class: Type[BotArchetype],
        data: Dict[str, pd.DataFrame],
        param_grid: Optional[Any] = None,
        in_sample_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calibrates a single bot archetype:
        1. Slices data chronologically into in-sample (67%) and out-of-sample (33%).
        2. Evaluates all parameter candidates simultaneously in an in-sample batch.
        3. Ranks combinations via composite in-sample fitness.
        4. Validates top K candidates on the out-of-sample window.
        5. Selects the best robust parameter set and evaluates full period performance.
        """
        sample_bot = bot_class()
        bot_id = sample_bot.bot_id
        persona_name = sample_bot.name

        # Resolve parameter candidates list
        if param_grid is None:
            candidates = DEFAULT_PERSONA_GRIDS.get(bot_id, [sample_bot.get_params()])
        elif isinstance(param_grid, dict):
            # Dict of lists -> Cartesian product
            candidates = generate_grid_combinations(param_grid)
        elif isinstance(param_grid, list):
            candidates = param_grid
        else:
            candidates = [sample_bot.get_params()]

        if not candidates:
            candidates = [sample_bot.get_params()]

        # 1. Chronological Train / Test Split
        split_pct = in_sample_pct or self.in_sample_pct
        train_data, val_data = self.split_data(data, in_sample_pct=split_pct)

        # 2. In-Sample Simultaneous Batch Simulation
        is_bots = []
        for idx, p in enumerate(candidates):
            b = bot_class(params=p)
            b.bot_id = f"{bot_id}_g{idx}"
            is_bots.append(b)

        is_backtest_res = self.engine.run_backtest(bots=is_bots, data=train_data)

        scored_candidates = []
        for idx, p in enumerate(candidates):
            sub_id = f"{bot_id}_g{idx}"
            bot_perf = is_backtest_res["bots"][sub_id]
            metrics = bot_perf["metrics"]
            fit = self.compute_fitness(metrics)
            scored_candidates.append({
                "idx": idx,
                "params": p,
                "is_metrics": metrics,
                "is_fitness": fit
            })

        # Rank candidates: primary = is_fitness, secondary = net_return_pct, tertiary = profit_factor
        def _rank_is(c: Dict[str, Any]) -> Tuple[float, float, float]:
            return (
                c["is_fitness"],
                float(c["is_metrics"].get("net_return_pct", 0.0)),
                float(c["is_metrics"].get("profit_factor", 0.0))
            )

        scored_candidates.sort(key=_rank_is, reverse=True)
        top_k = min(self.top_k_candidates, len(scored_candidates))
        selected_top = scored_candidates[:top_k]

        # 3. Out-of-Sample Batch Simulation for Top Candidates
        oos_bots = []
        for c in selected_top:
            b = bot_class(params=c["params"])
            b.bot_id = f"{bot_id}_oos_{c['idx']}"
            oos_bots.append(b)

        oos_backtest_res = self.engine.run_backtest(bots=oos_bots, data=val_data)

        # 4. Walk-Forward Robust Scoring & Selection
        best_candidate = None
        best_wf_score = -999999.0
        best_oos_metrics = {}

        for c in selected_top:
            sub_id = f"{bot_id}_oos_{c['idx']}"
            oos_metrics = oos_backtest_res["bots"][sub_id]["metrics"]
            wf_score = self.compute_walk_forward_score(c["is_metrics"], oos_metrics)

            if wf_score > best_wf_score or best_candidate is None:
                best_wf_score = wf_score
                best_candidate = c
                best_oos_metrics = oos_metrics

        if best_candidate is None:
            best_candidate = scored_candidates[0]
            best_oos_metrics = {}
            best_wf_score = best_candidate["is_fitness"]

        # 5. Full Period Evaluation for Final Calibrated Bot
        final_bot = bot_class(params=best_candidate["params"])
        final_bot.bot_id = bot_id
        full_res = self.engine.run_backtest(bots=[final_bot], data=data)
        full_metrics = full_res["bots"][bot_id]["metrics"]

        # Overfitting diagnosis
        is_ret = float(best_candidate["is_metrics"].get("net_return_pct", 0.0))
        oos_ret = float(best_oos_metrics.get("net_return_pct", 0.0))
        is_overfitted = bool(is_ret > 5.0 and oos_ret < -2.0)

        # Status categorization
        total_trades_is = int(best_candidate["is_metrics"].get("total_trades", 0))
        if total_trades_is < self.min_trades:
            status = "insufficient_trades"
        elif is_overfitted:
            status = "degraded"
        elif best_candidate["is_fitness"] > 0:
            status = "optimal"
        else:
            status = "active"

        # Determine dates for metadata
        first_sym = next(iter(data.keys()))
        t_start = str(train_data[first_sym].index[0])[:10] if train_data and len(train_data[first_sym]) > 0 else ""
        t_end = str(train_data[first_sym].index[-1])[:10] if train_data and len(train_data[first_sym]) > 0 else ""
        v_start = str(val_data[first_sym].index[0])[:10] if val_data and len(val_data[first_sym]) > 0 else ""
        v_end = str(val_data[first_sym].index[-1])[:10] if val_data and len(val_data[first_sym]) > 0 else ""

        result: Dict[str, Any] = {
            "bot_id": bot_id,
            "name": persona_name,
            "persona": sample_bot.persona,
            "persona_name": persona_name,
            "calibrated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": status,
            "best_params": best_candidate["params"],
            "in_sample_metrics": best_candidate["is_metrics"],
            "out_of_sample_metrics": best_oos_metrics,
            "full_period_metrics": full_metrics,
            "in_sample_fitness": best_candidate["is_fitness"],
            "out_of_sample_fitness": calculate_fitness_score(best_oos_metrics, min_trades=1),
            "walk_forward_score": best_wf_score,
            "is_overfitted": is_overfitted,
            "total_evaluated": len(candidates),
            "metadata": {
                "training_window": {
                    "in_sample_start": t_start,
                    "in_sample_end": t_end,
                    "out_of_sample_start": v_start,
                    "out_of_sample_end": v_end,
                    "split_ratio": split_pct
                },
                "fitness_score": best_candidate["is_fitness"],
                "candidates_evaluated": len(candidates),
                "selection_reason": "Highest walk-forward composite fitness"
            }
        }

        return _to_json_safe(result)


# ============================================================================
# MASTER ARENA CALIBRATION ENTRY POINT
# ============================================================================

def calibrate_all_personas(
    data: Optional[Dict[str, pd.DataFrame]] = None,
    years: int = 3,
    in_sample_pct: float = 0.67,
    custom_grids: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    offline: bool = True
) -> Dict[str, Any]:
    """
    Executes walk-forward calibration across all 5 bot personas:
    - ConservativeSwingBot
    - AggressiveBreakoutBot
    - ContrarianDipBuyerBot
    - FinBERTCatalystHunterBot
    - TrendFollowerBot

    Returns a standardized dictionary ready for Supabase persistence and CLI reporting.
    """
    start_time = time.time()

    # Load 3-year universe data if not supplied
    if data is None:
        fetcher = HistoricalDataFetcher(offline_only=offline)
        universe_symbols = ["^NSEI", "NIFTYBEES.NS", "NVDA", "AAPL", "MSFT", "SPY"]
        data = fetcher.fetch_universe_bars(symbols=universe_symbols, years=years)

    calibrator = WalkForwardCalibrator(in_sample_pct=in_sample_pct)

    persona_classes: List[Type[BotArchetype]] = [
        ConservativeSwingBot,
        AggressiveBreakoutBot,
        ContrarianDipBuyerBot,
        FinBERTCatalystHunterBot,
        TrendFollowerBot,
    ]

    personas_results: Dict[str, Dict[str, Any]] = {}
    grids = custom_grids or {}

    for bot_cls in persona_classes:
        sample_b = bot_cls()
        bot_id = sample_b.bot_id
        grid = grids.get(bot_id, DEFAULT_PERSONA_GRIDS.get(bot_id))
        res = calibrator.calibrate(bot_cls, data, param_grid=grid, in_sample_pct=in_sample_pct)
        personas_results[bot_id] = res

    elapsed = round(time.time() - start_time, 2)

    output: Dict[str, Any] = {
        "calibrated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "universe": list(data.keys()),
        "years": years,
        "status": "success",
        "execution_time_seconds": elapsed,
        "personas": personas_results,
    }

    # For maximum backwards and forwards compatibility, also map personas at the top level
    for bot_id, prof in personas_results.items():
        output[bot_id] = prof

    return _to_json_safe(output)
