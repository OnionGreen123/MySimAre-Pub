"""Shared types and lightweight schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Message:
    role: str
    content: str


@dataclass
class LLMCall:
    system_prompt: str
    user_prompt: str
    messages: List[Dict[str, str]]
    output: List[str]
    model_name: str
    temperature: float
    max_tokens: int
    n: int


@dataclass
class KnowledgeStateUpdate:
    previous_state: str
    new_state: str
    evidence: str
    confidence: Optional[float]

