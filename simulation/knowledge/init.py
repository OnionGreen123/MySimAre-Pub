"""Initialize dynamic knowledge states (K0)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..core.prompts import load_prompt
from .concept_graph import format_concept_list_with_prerequisites


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


async def initialize_dynamic_knowledge_states(
    *,
    problems: List[str],
    problem_ids: List[str],
    concept_graph: Dict[str, List[Dict[str, Any]]],
    education_levels: List[str],
    indicators: List[str],
    model_client: Any,
    max_tokens: int = 1200,
    show_progress: bool = True,
    prompt_path: str = "simulation/prompts/dynamic-knowledge-init.txt",
) -> List[Dict[str, Any]]:
    """
    Ported entry point for K0 initialization.
    Uses the dynamic-knowledge-init prompt and a single model client.
    """
    template = load_prompt(prompt_path)
    contexts = []
    for problem, problem_id, education_level, indicator in zip(
        problems, problem_ids, education_levels, indicators
    ):
        concepts = concept_graph.get(problem_id, [])
        concept_list = format_concept_list_with_prerequisites(concepts)
        prompt = template.format(
            math_problem=problem,
            concept_list_with_prerequisites=concept_list,
            education_level=education_level,
            indicators=indicator,
        )
        contexts.append(
            [
                {"role": "system", "content": "You are an expert educational diagnostician."},
                {"role": "user", "content": prompt},
            ]
        )
    responses = await model_client.generate_responses(
        contexts, temperature=0.7, max_tokens=max_tokens, n=1, show_progress=show_progress
    )
    parsed_states = []
    for response in responses:
        parsed = _extract_json_object(response[0] if response else "")
        if not parsed:
            parsed = {"_raw_response": response[0] if response else ""}
        parsed_states.append(parsed)
    return parsed_states

