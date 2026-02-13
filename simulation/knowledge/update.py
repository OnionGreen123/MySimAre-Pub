"""Update dynamic knowledge states (Kt)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..core.prompts import load_prompt
from .gating import clamp_state_by_prereqs


def _format_previous_states(concept_names: List[str], knowledge_state: Dict[str, Any]) -> str:
    if not knowledge_state:
        return "{}"
    formatted = {}
    for concept in concept_names:
        if concept in knowledge_state:
            formatted[concept] = {"state": knowledge_state[concept].get("state", "")}
    return json.dumps(formatted, indent=2)


def _format_prerequisite_states(
    concept_names: List[str],
    knowledge_state: Dict[str, Any],
    concept_graph: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    problem_id: Optional[str] = None,
) -> str:
    if not concept_graph or not problem_id:
        return "{}"
    concept_items = concept_graph.get(str(problem_id), []) or []
    prereq_map = {}
    for item in concept_items:
        concept_id = item.get("concept_id")
        if not concept_id or concept_id not in concept_names:
            continue
        prereqs = item.get("prerequisites", []) or []
        prereq_entries = []
        for prereq in prereqs:
            prereq_state = knowledge_state.get(prereq, {}).get("state", "unknown")
            prereq_entries.append({"concept": prereq, "state": prereq_state})
        prereq_map[concept_id] = prereq_entries
    return json.dumps(prereq_map, indent=2)


async def update_dynamic_knowledge_state(
    *,
    assistant_message: str,
    concept_names: List[str],
    knowledge_state: Dict[str, Any],
    user_response_analysis: str,
    model_client: Any,
    concept_graph: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    problem_id: Optional[str] = None,
    max_tokens: int = 1200,
    show_progress: bool = False,
    prompt_path: str = "simulation/prompts/dynamic-knowledge-update.txt",
) -> Dict[str, Any]:
    template = load_prompt(prompt_path)
    previous_states = _format_previous_states(concept_names, knowledge_state)
    prerequisite_states = _format_prerequisite_states(
        concept_names,
        knowledge_state,
        concept_graph=concept_graph,
        problem_id=problem_id,
    )
    extracted_concepts = json.dumps(concept_names, indent=2)
    prompt = template.format(
        assistant_message=assistant_message,
        extracted_concepts=extracted_concepts,
        previous_states=previous_states,
        prerequisite_states=prerequisite_states,
        user_response_analysis=user_response_analysis,
    )
    responses = await model_client.generate_responses(
        [[{"role": "system", "content": "You are a professional learning assessment analyst."},
          {"role": "user", "content": prompt}]],
        temperature=0.7,
        max_tokens=max_tokens,
        n=1,
        show_progress=show_progress,
    )
    raw = responses[0][0] if responses and responses[0] else ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return knowledge_state

    updated_state = dict(knowledge_state)
    for concept_name, update_info in parsed.items():
        if concept_name not in updated_state:
            updated_state[concept_name] = {}
        if isinstance(update_info, dict):
            new_state = update_info.get("new_state")
            if new_state:
                new_state = clamp_state_by_prereqs(
                    concept_name=concept_name,
                    proposed_state=new_state,
                    knowledge_state=updated_state,
                    concept_graph=concept_graph,
                    problem_id=problem_id,
                )
                updated_state[concept_name]["state"] = new_state
            updated_state[concept_name]["evidence"] = update_info.get("evidence", "")
            updated_state[concept_name]["confidence"] = update_info.get("confidence", None)
    return updated_state

