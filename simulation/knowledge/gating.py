"""Hard gating rules for knowledge updates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .state import STATE_ORDER


def clamp_state_by_prereqs(
    *,
    concept_name: str,
    proposed_state: str,
    knowledge_state: Dict[str, Any],
    concept_graph: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    problem_id: Optional[str] = None,
) -> str:
    """
    Hard-gate state upgrades based on prerequisite states.
    If any prerequisite is unknown_unknown or not_introduced,
    the target concept cannot exceed "struggling".
    """
    if not concept_graph or not problem_id or not proposed_state:
        return proposed_state

    concept_items = concept_graph.get(str(problem_id), []) or []
    prereqs = []
    for item in concept_items:
        if item.get("concept_id") == concept_name:
            prereqs = item.get("prerequisites", []) or []
            break

    if not prereqs:
        return proposed_state

    low_states = {"unknown_unknown", "not_introduced"}
    prereq_states = [
        knowledge_state.get(prereq, {}).get("state", "unknown_unknown")
        for prereq in prereqs
    ]
    if any(state in low_states for state in prereq_states):
        max_allowed = "struggling"
        try:
            if STATE_ORDER.index(proposed_state) > STATE_ORDER.index(max_allowed):
                return max_allowed
        except ValueError:
            return proposed_state

    return proposed_state

