"""Configuration helpers for the reconstructed simulation."""

from dataclasses import dataclass
import json
import os
from typing import Any, Dict


def _load_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from config.json."""

    log_llm_calls: bool = False
    log_llm_path: str = "logs/llm_calls.jsonl"
    print_llm_calls: bool = False

    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.environ.get("ANTHROPIC_KEY", "")
    gemini_api_key: str = os.environ.get("GEMINI_KEY", "")

    @staticmethod
    def from_config(path: str = "simulation/config.json") -> "Settings":
        data = _load_config(path)
        return Settings(
            log_llm_calls=bool(data.get("log_llm_calls", False)),
            log_llm_path=str(data.get("log_llm_path", "logs/llm_calls.jsonl")),
            print_llm_calls=bool(data.get("print_llm_calls", False)),
        )

