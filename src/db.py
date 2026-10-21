import os
import json
import time
import uuid
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Union
from supabase import create_client, Client
from src.config import SUPABASE_URL, SUPABASE_KEY

# Persistent local fallback storage paths
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR = DEFAULT_DATA_DIR
LOCAL_TRAINING_RUNS_FILE = DATA_DIR / "bot_training_runs.json"
TRAINING_RUNS_FILE = LOCAL_TRAINING_RUNS_FILE
LOCAL_CALIBRATED_PARAMS_FILE = DATA_DIR / "bot_calibrated_parameters.json"
CALIBRATED_PARAMS_FILE = LOCAL_CALIBRATED_PARAMS_FILE


def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in your .env file."
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def upsert_market_news(client: Optional[Client], articles_with_embeddings: List[Dict[str, Any]]) -> int:
    """
    Upserts articles and their vectors into Supabase.
    Uses 'link' as the conflict resolution key to prevent duplicate articles.
    Resilient to database disconnects and individual record errors.
    """
    if not articles_with_embeddings:
        return 0

    if not client:
        print("[Database] Warning: Cannot upsert articles because Supabase client is None.")
        return 0

    inserted_count = 0
    # Batch upsert or item-by-item with upsert
    for item in articles_with_embeddings:
        link = item.get("link")
        title = item.get("title")
        if not link or not title:
            continue

        record = {
            "title": title,
            "link": link,
            "source": item.get("source", "Unknown"),
            "published_at": item.get("published_at"),
            "summary": item.get("summary", ""),
            "content": item.get("content", ""),
            "embedding": item.get("embedding")
        }
        try:
            # ON CONFLICT (link) DO UPDATE SET embedding, summary, published_at
            client.table("market_news").upsert(record, on_conflict="link").execute()
            inserted_count += 1
        except Exception as e:
            print(f"[Database] Error upserting article '{title[:30]}...': {e}")

    return inserted_count


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return float(dot / ((norm_a ** 0.5) * (norm_b ** 0.5)))


def search_similar_news(
    client: Optional[Client],
    query_embedding: List[float],
    threshold: Optional[float] = None,
    count: Optional[int] = None,
    match_threshold: float = 0.3,
    match_count: int = 5,
    fallback_articles: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Calls the match_market_news RPC function in Supabase to perform cosine similarity search.
    If the Supabase client is None, network fails, or credentials are unavailable,
    logs a warning and provides an in-memory cosine similarity fallback against
    fallback_articles (if provided) or returns an empty list gracefully.
    """
    effective_threshold = threshold if threshold is not None else match_threshold
    effective_count = count if count is not None else match_count

    if client is not None:
        try:
            response = client.rpc(
                "match_market_news",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": effective_threshold,
                    "match_count": effective_count
                }
            ).execute()
            if response and hasattr(response, "data") and response.data is not None:
                return response.data
        except Exception as e:
            print(f"[Database] Warning: Supabase RPC error during similarity search: {e}")
    else:
        print("[Database] Warning: Supabase client is None. Proceeding with fallback if available.")

    # In-memory cosine similarity fallback
    if fallback_articles:
        results = []
        for art in fallback_articles:
            emb = art.get("embedding")
            if emb:
                sim = _cosine_similarity(query_embedding, emb)
                if sim >= effective_threshold:
                    art_copy = dict(art)
                    art_copy["similarity"] = sim
                    results.append(art_copy)
        results.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        return results[:effective_count]

    return []


# ============================================================================
# MILESTONE 3: SERIALIZATION & PERSISTENCE HELPERS
# ============================================================================

def _get_training_runs_file() -> Path:
    """Dynamically resolves the training runs file path, supporting test mocking."""
    import src.db as _mod
    default_file = DEFAULT_DATA_DIR / "bot_training_runs.json"
    if getattr(_mod, "TRAINING_RUNS_FILE", default_file) != default_file:
        return Path(_mod.TRAINING_RUNS_FILE)
    if getattr(_mod, "LOCAL_TRAINING_RUNS_FILE", default_file) != default_file:
        return Path(_mod.LOCAL_TRAINING_RUNS_FILE)
    if getattr(_mod, "DATA_DIR", DEFAULT_DATA_DIR) != DEFAULT_DATA_DIR:
        return Path(_mod.DATA_DIR) / "bot_training_runs.json"
    return default_file


def _get_calibrated_params_file() -> Path:
    """Dynamically resolves the calibrated params file path, supporting test mocking."""
    import src.db as _mod
    default_file = DEFAULT_DATA_DIR / "bot_calibrated_parameters.json"
    if getattr(_mod, "CALIBRATED_PARAMS_FILE", default_file) != default_file:
        return Path(_mod.CALIBRATED_PARAMS_FILE)
    if getattr(_mod, "LOCAL_CALIBRATED_PARAMS_FILE", default_file) != default_file:
        return Path(_mod.LOCAL_CALIBRATED_PARAMS_FILE)
    if getattr(_mod, "DATA_DIR", DEFAULT_DATA_DIR) != DEFAULT_DATA_DIR:
        return Path(_mod.DATA_DIR) / "bot_calibrated_parameters.json"
    return default_file


def _to_json_safe(obj: Any) -> Any:
    """Recursively converts numpy, pandas, and datetime objects into JSON-safe standard Python types."""
    try:
        import numpy as np
    except ImportError:
        np = None
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    elif np is not None and isinstance(obj, (np.floating, float)):
        return round(float(obj), 4) if math.isfinite(obj) else 0.0
    elif isinstance(obj, float):
        return round(float(obj), 4) if math.isfinite(obj) else 0.0
    elif np is not None and isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, int):
        return int(obj)
    elif np is not None and isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, bool):
        return bool(obj)
    elif pd is not None and isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _atomic_write_json(file_path: Union[str, Path], data: Any):
    """Crash-proof atomic JSON serialization using temporary file and fsync."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_data = _to_json_safe(data)
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise exc


def _safe_read_json(file_path: Union[str, Path], default: Any = None) -> Any:
    """Safely loads a JSON file, returning default if missing or unreadable."""
    path = Path(file_path)
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[Storage] Warning: Could not parse {path.name} ({exc}), returning fallback default.")
        return default


# ============================================================================
# MILESTONE 3: SUPABASE & OFFLINE PERSISTENCE FUNCTIONS
# ============================================================================

def save_training_run(
    client: Optional[Client],
    run_data: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Persists a historical training run record.
    Dual-writes to local cache ('data/bot_training_runs.json') and attempts Supabase persistence.
    Returns (success: bool, error_message: Optional[str]).
    """
    if not isinstance(run_data, dict):
        return False, "Invalid run_data: expected a dictionary"

    now_iso = datetime.now(timezone.utc).isoformat()
    clean_run = _to_json_safe(dict(run_data))

    # Standardize metadata keys
    if "created_at" not in clean_run and "calibrated_at" in clean_run:
        clean_run["created_at"] = clean_run["calibrated_at"]
    elif "created_at" not in clean_run:
        clean_run["created_at"] = now_iso

    if "calibrated_at" not in clean_run:
        clean_run["calibrated_at"] = clean_run["created_at"]

    if "run_id" not in clean_run:
        clean_run["run_id"] = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    if "years" in clean_run and "years_trained" not in clean_run:
        clean_run["years_trained"] = clean_run["years"]
    elif "years_trained" in clean_run and "years" not in clean_run:
        clean_run["years"] = clean_run["years_trained"]
    elif "years" not in clean_run:
        clean_run["years"] = 3
        clean_run["years_trained"] = 3

    if "universe" in clean_run and "universe_symbols" not in clean_run:
        clean_run["universe_symbols"] = clean_run["universe"]
    elif "universe_symbols" in clean_run and "universe" not in clean_run:
        clean_run["universe"] = clean_run["universe_symbols"]

    if "status" not in clean_run:
        clean_run["status"] = "success"

    # 1. Local JSON write (always guaranteed)
    try:
        runs_file = _get_training_runs_file()
        existing_runs = _safe_read_json(runs_file, default=[])
        if not isinstance(existing_runs, list):
            existing_runs = []
        runs_list = [r for r in existing_runs if isinstance(r, dict) and r.get("run_id") != clean_run["run_id"]]
        runs_list.insert(0, clean_run)
        _atomic_write_json(runs_file, runs_list[:50])
    except Exception as exc:
        err_msg = f"Failed to save training run locally: {exc}"
        print(f"[Database] Error: {err_msg}")
        return False, err_msg

    # 2. Supabase Cloud write (best-effort)
    if client is not None:
        try:
            years_val = clean_run.get("years_trained", 3)
            try:
                years_trained_num = int(years_val) if float(years_val).is_integer() else float(years_val)
            except (ValueError, TypeError):
                years_trained_num = 3

            record = {
                "run_id": str(clean_run["run_id"]),
                "created_at": str(clean_run["created_at"]),
                "years_trained": years_trained_num,
                "universe_symbols": list(clean_run.get("universe_symbols", [])),
                "bot_archetypes": list(clean_run.get("bot_archetypes", list(clean_run.get("personas", {}).keys()))),
                "training_metrics": clean_run.get("training_metrics", clean_run.get("personas", {})),
                "profit_factor": float(clean_run.get("profit_factor", 0.0)),
                "win_rate_pct": float(clean_run.get("win_rate_pct", 0.0)),
                "max_drawdown_pct": float(clean_run.get("max_drawdown_pct", 0.0)),
                "net_return_pct": float(clean_run.get("net_return_pct", 0.0)),
                "cagr_pct": float(clean_run.get("cagr_pct", 0.0)),
                "total_trades": int(clean_run.get("total_trades", 0)),
                "execution_time_sec": float(clean_run.get("execution_time_sec", clean_run.get("execution_time_seconds", 0.0))),
                "in_sample_window": clean_run.get("in_sample_window", {}),
                "out_of_sample_window": clean_run.get("out_of_sample_window", {}),
                "status": str(clean_run["status"])
            }
            client.table("bot_training_runs").insert(record).execute()
            print("✅ [Database] Successfully persisted training run to Supabase.")
            return True, None
        except Exception as exc:
            print(f"[Database] Warning: Supabase insert into bot_training_runs failed ({exc}). Persisted locally.")
            return True, None
    else:
        print(f"[Database] Client is None. Persisted training run locally to {_get_training_runs_file().name}.")
        return True, None


def save_calibrated_parameters(
    client: Optional[Client],
    bot_id: str,
    parameters: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
    *,
    in_sample_metrics: Optional[Dict[str, Any]] = None,
    out_of_sample_metrics: Optional[Dict[str, Any]] = None,
    full_metrics: Optional[Dict[str, Any]] = None,
    active: bool = True,
    status: str = "active",
    training_run_id: Optional[str] = None,
    **kwargs: Any
) -> Tuple[bool, Optional[str]]:
    """
    Persists optimal parameters for a bot persona.
    Dual-writes to local cache ('data/bot_calibrated_parameters.json') and attempts Supabase persistence.
    Returns (success: bool, error_message: Optional[str]).
    """
    if not bot_id or not isinstance(bot_id, str):
        return False, "Invalid bot_id: expected non-empty string"
    if not isinstance(parameters, dict):
        return False, "Invalid parameters: expected a dictionary"

    now_iso = datetime.now(timezone.utc).isoformat()
    clean_params = _to_json_safe(parameters)
    norm_metrics = metrics if isinstance(metrics, dict) else {}
    clean_metrics = _to_json_safe(norm_metrics)
    clean_in_sample = _to_json_safe(in_sample_metrics) if isinstance(in_sample_metrics, dict) else clean_metrics
    clean_out_sample = _to_json_safe(out_of_sample_metrics) if isinstance(out_of_sample_metrics, dict) else {}
    clean_full = _to_json_safe(full_metrics) if isinstance(full_metrics, dict) else clean_metrics

    profile_entry = {
        "bot_id": bot_id,
        "calibrated_at": now_iso,
        "parameters": clean_params,
        "metrics": clean_metrics,
        "in_sample_metrics": clean_in_sample,
        "out_of_sample_metrics": clean_out_sample,
        "full_metrics": clean_full,
        "status": status,
        "active": bool(active),
        "updated_at": now_iso
    }
    if training_run_id:
        profile_entry["training_run_id"] = str(training_run_id)

    # 1. Local JSON write (always guaranteed)
    try:
        cache_file = _get_calibrated_params_file()
        cache = _safe_read_json(cache_file, default={})
        if not isinstance(cache, dict):
            cache = {}
        cache[bot_id] = profile_entry
        _atomic_write_json(cache_file, cache)
    except Exception as exc:
        err_msg = f"Failed to save calibrated parameters locally for {bot_id}: {exc}"
        print(f"[Database] Error: {err_msg}")
        return False, err_msg

    # 2. Supabase Cloud write (best-effort)
    if client is not None:
        try:
            # Set older records for this bot to inactive
            try:
                client.table("bot_calibrated_parameters").update({"active": False}).eq("bot_id", bot_id).execute()
            except Exception:
                pass

            record = {
                "bot_id": bot_id,
                "calibrated_at": now_iso,
                "parameters": clean_params,
                "in_sample_metrics": clean_in_sample,
                "out_of_sample_metrics": clean_out_sample,
                "full_metrics": clean_full,
                "active": bool(active),
                "updated_at": now_iso
            }
            if training_run_id:
                record["training_run_id"] = str(training_run_id)

            try:
                client.table("bot_calibrated_parameters").insert(record).execute()
            except Exception:
                client.table("bot_calibrated_parameters").upsert(record, on_conflict="bot_id").execute()

            print(f"✅ [Database] Successfully persisted calibrated parameters for '{bot_id}' to Supabase.")
            return True, None
        except Exception as exc:
            print(f"[Database] Warning: Supabase save failed for '{bot_id}' ({exc}). Persisted locally.")
            return True, None
    else:
        print(f"[Database] Client is None. Persisted parameters for '{bot_id}' locally to {_get_calibrated_params_file().name}.")
        return True, None


def load_calibrated_parameters(
    client: Optional[Client],
    bot_id: str
) -> Optional[Dict[str, Any]]:
    """
    3-Tier Parameter Resolution Architecture:
    Tier 1: Supabase Cloud query ('bot_calibrated_parameters' where bot_id=bot_id and active=true)
    Tier 2: Local JSON cache ('data/bot_calibrated_parameters.json')
    Tier 3: None (signals caller to fall back to hardcoded BotArchetype constructor defaults)
    """
    if not bot_id:
        return None

    # Tier 1: Supabase Cloud
    if client is not None:
        try:
            res = (
                client.table("bot_calibrated_parameters")
                .select("*")
                .eq("bot_id", bot_id)
                .eq("active", True)
                .order("calibrated_at", desc=True)
                .limit(1)
                .execute()
            )
            if res and hasattr(res, "data") and res.data and len(res.data) > 0:
                row = res.data[0]
                raw_params = row.get("parameters")
                if isinstance(raw_params, dict) and raw_params:
                    # Update local cache mirror preserving existing metadata such as status
                    try:
                        cache_file = _get_calibrated_params_file()
                        cache = _safe_read_json(cache_file, default={})
                        if not isinstance(cache, dict):
                            cache = {}
                        existing_entry = cache.get(bot_id, {})
                        merged_entry = dict(existing_entry) if isinstance(existing_entry, dict) else {}
                        merged_entry.update(row)
                        if "status" not in merged_entry or not merged_entry["status"]:
                            try:
                                from src.calibration import check_overfitting_gate
                                is_m = merged_entry.get("in_sample_metrics") or {}
                                oos = merged_entry.get("out_of_sample_metrics") or {}
                                is_overfit, _ = check_overfitting_gate(is_m, oos)
                                merged_entry["status"] = "degraded" if is_overfit else "optimal"
                            except Exception:
                                oos = merged_entry.get("out_of_sample_metrics") or {}
                                oos_ret = float(oos.get("net_return_pct", 0.0))
                                oos_pf = float(oos.get("profit_factor", 0.0))
                                if oos_ret <= 0.0 or oos_pf < 1.0:
                                    merged_entry["status"] = "degraded"
                                else:
                                    merged_entry["status"] = "optimal"
                        cache[bot_id] = merged_entry
                        _atomic_write_json(cache_file, cache)
                    except Exception:
                        pass

                    result = dict(raw_params)
                    result["parameters"] = raw_params
                    result["bot_id"] = bot_id
                    for m in ("metrics", "in_sample_metrics", "out_of_sample_metrics", "full_metrics"):
                        if m in row:
                            result[m] = row[m]
                    print(f"✅ [Parameters] Loaded calibrated parameters for '{bot_id}' from Supabase Cloud.")
                    return result
        except Exception as exc:
            print(f"[Database] Warning: Supabase error loading parameters for '{bot_id}': {exc}. Falling back to Tier 2 (Local JSON).")

    # Tier 2: Local JSON Cache
    try:
        cache_file = _get_calibrated_params_file()
        cache = _safe_read_json(cache_file, default=None)
        if isinstance(cache, dict):
            entry = None
            if bot_id in cache:
                entry = cache[bot_id]
            elif "bots" in cache and isinstance(cache["bots"], dict) and bot_id in cache["bots"]:
                entry = cache["bots"][bot_id]
            elif "personas" in cache and isinstance(cache["personas"], dict) and bot_id in cache["personas"]:
                entry = cache["personas"][bot_id]

            if isinstance(entry, dict):
                raw_params = entry.get("parameters") or entry.get("best_params")
                if isinstance(raw_params, dict) and raw_params:
                    result = dict(raw_params)
                    result["parameters"] = raw_params
                    result["bot_id"] = bot_id
                    for m in ("metrics", "in_sample_metrics", "out_of_sample_metrics", "full_metrics"):
                        if m in entry:
                            result[m] = entry[m]
                    print(f"ℹ️ [Parameters] Loaded calibrated parameters for '{bot_id}' from local JSON cache.")
                    return result
                elif any(k in entry for k in ("stop_loss_pct", "take_profit_pct", "allocation_pct", "max_positions", "min_momentum_5d")):
                    result = dict(entry)
                    result["parameters"] = dict(entry)
                    result["bot_id"] = bot_id
                    print(f"ℹ️ [Parameters] Loaded calibrated parameters for '{bot_id}' from local JSON cache.")
                    return result
    except Exception as exc:
        print(f"[Database] Warning: Failed to load local calibrated parameters for '{bot_id}': {exc}")

    # Tier 3: None (signals caller to use hardcoded BotArchetype defaults)
    print(f"ℹ️ [Parameters] No cloud or local calibration for '{bot_id}'. Using hardcoded persona defaults.")
    return None

