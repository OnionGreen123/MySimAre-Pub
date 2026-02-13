"""IU-based knowledge state initialization (v3)."""

from __future__ import annotations

import random
from typing import Dict, List


def _topological_sort(nodes: List[Dict[str, str]], edges: List[Dict[str, str]]) -> List[str]:
    incoming = {n["id"]: 0 for n in nodes}
    children = {n["id"]: [] for n in nodes}
    for e in edges:
        src = e.get("from")
        dst = e.get("to")
        if src in incoming and dst in incoming:
            incoming[dst] += 1
            children[src].append(dst)
    queue = [n_id for n_id, deg in incoming.items() if deg == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in children.get(node, []):
            incoming[child] -= 1
            if incoming[child] == 0:
                queue.append(child)
    return order


def _get_prereqs(edges: List[Dict[str, str]], node_id: str) -> List[str]:
    return [e["from"] for e in edges if e.get("to") == node_id]


def initialize_knowledge_state(
    iu_graph: Dict[str, List[Dict[str, str]]],
    level: str,
    rng: random.Random,
) -> Dict[str, List[str]]:
    nodes = iu_graph.get("nodes", [])
    edges = iu_graph.get("edges", [])
    all_nodes = _topological_sort(nodes, edges)
    total = len(all_nodes)

    if level == "novice":
        target_known_ratio = rng.uniform(0.0, 0.1)
        target_partial_ratio = rng.uniform(0.05, 0.15)
    elif level == "intermediate":
        target_known_ratio = rng.uniform(0.2, 0.4)
        target_partial_ratio = rng.uniform(0.1, 0.2)
    elif level == "advanced":
        target_known_ratio = rng.uniform(0.5, 0.7)
        target_partial_ratio = rng.uniform(0.1, 0.2)
    else:
        raise ValueError(f"Unsupported level: {level}")

    known = set()
    partially_known = set()
    unknown = set(all_nodes)

    target_known_count = int(target_known_ratio * total)
    candidates = list(all_nodes)

    while len(known) < target_known_count and candidates:
        node = candidates.pop(0)
        prereqs = _get_prereqs(edges, node)
        if all(p in known for p in prereqs):
            if rng.random() < 0.7:
                known.add(node)
                unknown.discard(node)

    target_partial_count = int(target_partial_ratio * total)
    for node in list(unknown):
        if len(partially_known) >= target_partial_count:
            break
        prereqs = _get_prereqs(edges, node)
        known_prereq_ratio = len([p for p in prereqs if p in known]) / max(len(prereqs), 1)
        if known_prereq_ratio > 0:
            if rng.random() < 0.5:
                partially_known.add(node)
                unknown.discard(node)

    return {
        "known": list(known),
        "partially_known": list(partially_known),
        "unknown": list(unknown),
    }

