"""Dataset loaders for annotations and profiles."""

from __future__ import annotations

import json
import csv
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_annotations(path: str) -> List[Dict[str, Any]]:
    return load_json(path)


def load_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

