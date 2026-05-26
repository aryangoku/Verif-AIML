# core/utils.py

from typing import Any, Dict, Mapping


def safe_dict(value: Any) -> Dict:
    """Return value if it is a dict, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def nested_get(mapping: Any, *keys: str, default=None):
    """Safely traverse nested mappings; never call .get on None."""
    current = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key, default)
    return current


def sanitize_metrics(metrics: Any) -> Dict:
    """Convert numpy scalars to plain Python types for UI / JSON."""
    if not isinstance(metrics, dict):
        return {}

    clean = {}
    for key, value in metrics.items():
        if key == "all_model_results" and isinstance(value, dict):
            clean[key] = {
                model: {
                    stat: float(val) if hasattr(val, "item") else val
                    for stat, val in scores.items()
                }
                if isinstance(scores, dict)
                else scores
                for model, scores in value.items()
            }
        elif hasattr(value, "item"):
            clean[key] = float(value.item())
        elif isinstance(value, (int, float, str)):
            clean[key] = value
        else:
            clean[key] = value
    return clean
