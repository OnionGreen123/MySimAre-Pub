"""Profile feature filtering helpers."""

from __future__ import annotations

from typing import Any, Dict, List


def filter_profile_features(features: List[Dict[str, Any]], allowed_names: List[str]) -> List[Dict[str, Any]]:
    if not allowed_names:
        return features
    allowed = set(allowed_names)
    return [f for f in features if f.get("Feature Name") in allowed]

