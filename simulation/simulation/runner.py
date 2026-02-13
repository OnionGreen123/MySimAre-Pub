"""CLI entry point for the reconstructed simulator."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import List, Dict

from ..core.models import SingleModelClient
from ..core.prompts import load_prompt
from ..data.loaders import load_annotations, load_json, load_csv_rows
from ..knowledge.init import initialize_dynamic_knowledge_states
from ..knowledge.iu_extraction import extract_iu_graph
from ..knowledge.iu_graph import build_concept_graph_from_iu
from ..knowledge.iu_init import initialize_knowledge_state
from ..profiles.interaction import format_interaction_profile
from .conversation import run_conversation_with_interaction_profile
from .length_control import count_words, round_down_to_nearest_5, round_up_to_nearest_5



def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstructed math tutoring simulator.")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--annotation_id", type=str, default="math_tutoring_annotations")
    parser.add_argument("--num_conversations", type=int, default=-1)
    parser.add_argument("--user_model", type=str, default="gpt-5-mini")
    parser.add_argument("--assistant_model", type=str, default="")
    parser.add_argument("--iu_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--length_control", action="store_true")
    parser.add_argument("--length_control_setting", type=str, default="range")
    parser.add_argument("--dynamic_knowledge_state_init", action="store_true")
    parser.add_argument("--data_root", type=str, default=r"D:\MySimAre\Data")
    parser.add_argument("--prompts_root", type=str, default="simulation/prompts")
    parser.add_argument("--input_csv", type=str, default=r"D:\MySimAre\Data\competition_math\data\train_fixed_level_4_5.csv")
    parser.add_argument("--knowledge_level", type=str, default="intermediate", choices=["novice", "intermediate", "advanced"])
    parser.add_argument("--seed", type=int, default=2)
    return parser


def _load_prompt_pair(prompts_root: str, version: str) -> tuple[str, str]:
    prompt_path = os.path.join(prompts_root, f"{version}.txt")
    initial_path = os.path.join(prompts_root, f"{version}-initial-query.txt")
    return load_prompt(prompt_path), load_prompt(initial_path)


def _build_length_control_list(
    annotations: List[dict],
    length_control_setting: str,
) -> List[str]:
    length_control_list = []
    for ann in annotations:
        user_queries = ann["user_queries"]
        problem_turns = ann["problem_1_turns"] if ann["problem_1_turns"] > 0 else len(user_queries)
        user_queries = user_queries[:problem_turns]
        query_length_list = [count_words(query) for query in user_queries]
        if length_control_setting == "range":
            rounded_min = int(round_down_to_nearest_5(min(query_length_list)))
            rounded_max = int(round_up_to_nearest_5(max(query_length_list)))
            length_text = f"between {rounded_min} and {rounded_max} words"
        elif length_control_setting == "average":
            avg_length = sum(query_length_list) / len(query_length_list)
            length_text = f"around {int(round_up_to_nearest_5(avg_length))} words"
        else:
            raise ValueError(f"Unsupported length_control_setting: {length_control_setting}")
        length_control_list.append(length_text)
    return length_control_list


async def main() -> None:
    parser = cli_parser()
    args = parser.parse_args()

    prompt_template, prompt_initial_query_template = _load_prompt_pair(args.prompts_root, args.version)

    # Load problems from CSV (question + reference answer)
    rows = load_csv_rows(args.input_csv)
    if args.num_conversations > 0:
        rows = rows[:args.num_conversations]

    annotations: List[Dict[str, str]] = []
    for idx, row in enumerate(rows):
        annotations.append(
            {
                "problem_id": str(idx),
                "question": row.get("problem", ""),
                "solution": row.get("solution", ""),
                "level": row.get("level", ""),
                "type": row.get("type", ""),
                "user_id": "csv_user",
                "model": args.assistant_model or args.user_model,
            }
        )

    if args.num_conversations > 0:
        annotations = annotations[:args.num_conversations]

    user_model_client = SingleModelClient(args.user_model)
    iu_model_client = SingleModelClient(args.iu_model)
    assistant_model_name = args.assistant_model or args.user_model
    assistant_model_client = SingleModelClient(assistant_model_name)

    # Build IU graphs from question + answer, then convert to concept graph
    iu_graphs: Dict[str, Dict] = {}
    for ann in annotations:
        iu_graph = await extract_iu_graph(
            question=ann["question"],
            answer=ann["solution"],
            model_client=iu_model_client,
            max_tokens=1200,
            show_progress=False,
            prompt_path=os.path.join(args.prompts_root, "iu_graph_extraction.txt"),
        )
        iu_graphs[str(ann["problem_id"])] = iu_graph

    concept_graph, id_maps = build_concept_graph_from_iu(iu_graphs)

    problems = [ann["question"] for ann in annotations]
    problem_ids = [str(ann["problem_id"]) for ann in annotations]

    length_control_list = []
    if args.length_control:
        length_control_list = _build_length_control_list(annotations, args.length_control_setting)

    # Build interaction-only user profiles (fallback if no profile file is available)
    user_profiles = []
    for ann, length_text in zip(annotations, length_control_list or ["" for _ in annotations]):
        user_profiles.append(format_interaction_profile([], length_text or "around 20 words"))

    knowledge_states = None
    if args.dynamic_knowledge_state_init:
        rng = random.Random(args.seed)
        knowledge_states = []
        for pid in problem_ids:
            iu_graph = iu_graphs.get(str(pid), {})
            id_map = id_maps.get(str(pid), {})
            edges = iu_graph.get("edges", [])
            prereqs_by_id: Dict[str, List[str]] = {}
            for e in edges:
                src = e.get("from")
                dst = e.get("to")
                if not src or not dst:
                    continue
                prereqs_by_id.setdefault(dst, []).append(src)

            state = initialize_knowledge_state(iu_graph, args.knowledge_level, rng)
            known_ids = set(state.get("known", []))
            partially_ids = set(state.get("partially_known", []))
            unknown_ids = set(state.get("unknown", []))

            mapped: Dict[str, Dict[str, str]] = {}
            for iu_id in known_ids:
                mapped[id_map.get(iu_id, iu_id)] = {"state": "knows_well"}

            for iu_id in partially_ids:
                prereqs = prereqs_by_id.get(iu_id, [])
                known_ratio = (
                    len([p for p in prereqs if p in known_ids]) / max(len(prereqs), 1)
                )
                state_label = "partial_understanding" if known_ratio >= 0.5 else "struggling"
                mapped[id_map.get(iu_id, iu_id)] = {"state": state_label}

            for iu_id in unknown_ids:
                prereqs = prereqs_by_id.get(iu_id, [])
                has_known_prereq = any(p in known_ids for p in prereqs)
                state_label = "not_introduced" if has_known_prereq else "unknown_unknown"
                mapped[id_map.get(iu_id, iu_id)] = {"state": state_label}

            knowledge_states.append(mapped)

    results = await run_conversation_with_interaction_profile(
        problems=problems,
        problem_ids=problem_ids,
        user_profiles=user_profiles,
        user_model_client=user_model_client,
        assistant_model_client=assistant_model_client,
        prompt_initial_query_template=prompt_initial_query_template,
        prompt_template=prompt_template,
        concept_graph=concept_graph,
        knowledge_states=knowledge_states,
        user_temperature=0.7,
        assistant_temperature=0.0,
        max_tokens=3000,
        max_turns=15,
        length_control_bool=args.length_control,
        length_control_list=length_control_list,
        show_progress=True,
    )

    output_dir = os.path.join("output", "competition_math", assistant_model_name)
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"{args.version}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

