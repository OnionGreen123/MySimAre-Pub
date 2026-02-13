"""Extract explained concepts from the tutor's last message."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ..core.prompts import load_prompt


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


async def extract_explained_concepts(
    *,
    assistant_message: str,
    candidate_concepts: List[str],
    model_client: Any,
    max_tokens: int = 600,
    show_progress: bool = False,
    prompt_path: str = "simulation/prompts/dynamic-knowledge-extract.txt",
) -> List[str]:
    template = load_prompt(prompt_path)
    prompt = template.format(
        assistant_message=assistant_message,
        candidate_concepts=json.dumps(candidate_concepts, indent=2),
    )
    responses = await model_client.generate_responses(
        [[{"role": "system", "content": "You are a professional concept extraction specialist."},
          {"role": "user", "content": prompt}]],
        temperature=0.3,
        max_tokens=max_tokens,
        n=1,
        show_progress=show_progress,
    )
    raw = responses[0][0] if responses and responses[0] else ""
    parsed = _extract_json_object(raw)
    concepts = parsed.get("explained_concepts", []) if isinstance(parsed, dict) else []
    if not isinstance(concepts, list):
        return []
    return [c for c in concepts if c in candidate_concepts]

