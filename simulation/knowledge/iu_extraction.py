"""IU graph extraction from question + reference answer."""

from __future__ import annotations

import json
import re
from typing import Any, Dict

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


async def extract_iu_graph(
    *,
    question: str,
    answer: str,
    model_client: Any,
    max_tokens: int = 1200,
    show_progress: bool = False,
    prompt_path: str = "prompts/iu_graph_extraction.txt",
) -> Dict[str, Any]:
    template = load_prompt(prompt_path)
    prompt = template.format(question=question, answer=answer)

    async def _call_once(prompt_text: str, temperature: float) -> str:
        responses = await model_client.generate_responses(
            [[{"role": "system", "content": "You are an expert knowledge graph extractor."},
              {"role": "user", "content": prompt_text}]],
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            show_progress=show_progress,
            json_mode=True,
        )
        return responses[0][0] if responses and responses[0] else ""

    raw = await _call_once(prompt, temperature=0.2)
    parsed = _extract_json_object(raw)
    if parsed:
        return parsed

    # Retry once with a stricter JSON-only instruction and lower temperature.
    retry_prompt = prompt + "\n\nReturn only valid JSON. Do not include any extra text."
    raw_retry = await _call_once(retry_prompt, temperature=0.0)
    parsed_retry = _extract_json_object(raw_retry)
    if parsed_retry:
        return parsed_retry

    raise RuntimeError("IU graph extraction failed: empty or unparsable response.")

