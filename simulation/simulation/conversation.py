"""Conversation loop orchestration (math tutoring only)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _format_knowledge_state(knowledge_state: Optional[Dict[str, Any]]) -> str:
    if not knowledge_state:
        return "{}"
    compact = {
        concept: {"state": info.get("state")}
        for concept, info in knowledge_state.items()
    }
    return json.dumps(compact, indent=2)


def _get_unknown_unknown_concepts(knowledge_state: Optional[Dict[str, Any]]) -> List[str]:
    if not knowledge_state:
        return []
    return [
        concept for concept, info in knowledge_state.items()
        if info.get("state") == "unknown_unknown"
    ]


def _get_askable_concepts(knowledge_state: Optional[Dict[str, Any]]) -> List[str]:
    if not knowledge_state:
        return []
    return [
        concept for concept, info in knowledge_state.items()
        if info.get("state") != "unknown_unknown"
    ]


def _get_last_assistant_message(assistant_messages: List[Dict[str, str]]) -> str:
    for message in reversed(assistant_messages):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def _is_stuck(knowledge_state_history: List[Dict[str, Any]]) -> bool:
    if not knowledge_state_history or len(knowledge_state_history) < 2:
        return False
    last_state = knowledge_state_history[-1] or {}
    prev_state = knowledge_state_history[-2] or {}
    return json.dumps(last_state, sort_keys=True) == json.dumps(prev_state, sort_keys=True)


def _get_misguided_attempt_hint(
    knowledge_state: Optional[Dict[str, Any]],
    knowledge_state_history: List[Dict[str, Any]],
) -> str:
    if not knowledge_state:
        return ""
    unknown_unknowns = _get_unknown_unknown_concepts(knowledge_state)
    if not unknown_unknowns:
        return ""
    if _is_stuck(knowledge_state_history):
        return "If you are stuck, attempt a misguided approach without naming the missing concept."
    return ""


async def run_conversation_batch(
    *,
    problems: List[str],
    user_model_client: Any,
    assistant_model_client: Any,
    prompt_initial_query_template: str,
    prompt_template: str,
    show_progress: bool = True,
    user_temperature: float = 0.7,
    assistant_temperature: float = 0.0,
    max_tokens: int = 3000,
    max_turns: int = 15,
    length_control_bool: bool = False,
    length_control_list: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Ported from utils.simulate_conversation_in_batch_math_tutoring (no refinement/user profile).
    """
    length_control_list = length_control_list or []

    conversations_data = []
    for i, problem in enumerate(problems):
        length_control = length_control_list[i] if length_control_bool else None
        data = {
            "problem": problem,
            "conversation": [],
            "conversation_history": "",
            "length_control": length_control,
            "assistant_messages": [],
            "first_query": True,
            "turns": 0,
            "finished": False,
            "over_max": False,
        }
        assistant_system_prompt = {
            "role": "system",
            "content": (
                "You are a skilled math tutor. Your goal is to help students understand and "
                "solve problems independently. Provide guidance based on their questions or "
                "mistakes. Ask questions to encourage their thinking and let students do most "
                "of the work themselves. Never give out the solution directly to students."
            ),
        }
        data["assistant_messages"].append(assistant_system_prompt)
        conversations_data.append(data)

    for turn in range(max_turns):
        user_full_contexts = []
        active_conversations = []
        for data in conversations_data:
            if data["finished"] or data["over_max"]:
                continue
            if data["first_query"]:
                if length_control_bool:
                    user_message_content = prompt_initial_query_template.format(
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                        length_control=data["length_control"],
                    )
                else:
                    user_message_content = prompt_initial_query_template.format(
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                    )
                data["first_query"] = False
            else:
                if length_control_bool:
                    user_message_content = prompt_template.format(
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                        length_control=data["length_control"],
                    )
                else:
                    user_message_content = prompt_template.format(
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                    )

            user_messages = [{"role": "user", "content": user_message_content}]
            data["user_messages"] = user_messages
            user_full_contexts.append(user_messages)
            active_conversations.append(data)

        if not active_conversations:
            break

        user_queries = await user_model_client.generate_responses(
            user_full_contexts,
            temperature=user_temperature,
            max_tokens=max_tokens,
            show_progress=show_progress,
        )

        for data, user_query in zip(active_conversations, user_queries):
            user_query_text = user_query[0] if user_query else ""
            if user_query_text.strip().lower() == "terminate: true":
                data["finished"] = True
                continue
            data["conversation"].append(("user", user_query_text))

            if "Thought:" in user_query_text:
                try:
                    if "Response:" in user_query_text:
                        query = user_query_text.split("Response:")[1].strip()
                    elif "Query:" in user_query_text:
                        query = user_query_text.split("Query:")[1].strip()
                    else:
                        query = user_query_text.split("Message:")[1].strip()
                except Exception:
                    data["finished"] = True
                    continue
            else:
                query = user_query_text

            if not user_query_text:
                data["finished"] = True
                continue

            if len(data["assistant_messages"]) == 1:
                first_turn_user_query = (
                    f"Here is the problem that you will tutor me on:\n"
                    f"{data['problem'].strip()}\n\n{query}"
                )
                data["assistant_messages"].append({"role": "user", "content": first_turn_user_query})
                data["first_query_content"] = query
            else:
                data["assistant_messages"].append({"role": "user", "content": query})

        active_conversations = [data for data in active_conversations if not data["finished"]]
        if not active_conversations:
            break

        assistant_full_contexts = [data["assistant_messages"] for data in active_conversations]
        assistant_responses = await assistant_model_client.generate_responses(
            assistant_full_contexts,
            temperature=assistant_temperature,
            max_tokens=max_tokens,
            show_progress=show_progress,
        )

        for data, assistant_response in zip(active_conversations, assistant_responses):
            assistant_text = assistant_response[0] if assistant_response else ""
            data["conversation"].append(("assistant", assistant_text))
            last_user_message = data["assistant_messages"][-1]["content"]
            if len(data["assistant_messages"]) == 2:
                last_user_message = data["first_query_content"]
            data["conversation_history"] += f"- You: {last_user_message}\n- AI Tutor: {assistant_text}\n"
            data["assistant_messages"].append({"role": "assistant", "content": assistant_text})

            data["turns"] += 1
            if not assistant_text:
                data["finished"] = True
            if data["turns"] >= max_turns:
                data["over_max"] = True

    return conversations_data


async def run_conversation_with_interaction_profile(
    *,
    problems: List[str],
    problem_ids: List[str],
    user_profiles: List[str],
    user_model_client: Any,
    assistant_model_client: Any,
    prompt_initial_query_template: str,
    prompt_template: str,
    concept_graph: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    knowledge_states: Optional[List[Dict[str, Any]]] = None,
    user_temperature: float = 0.7,
    assistant_temperature: float = 0.0,
    max_tokens: int = 3000,
    max_turns: int = 15,
    length_control_bool: bool = False,
    length_control_list: Optional[List[str]] = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Ported from utils.simulate_conversation_with_user_profile_in_batch_math_tutoring,
    simplified to interaction-style profiles only.
    """
    length_control_list = length_control_list or []

    conversations_data = []
    for i, problem in enumerate(problems):
        length_control = length_control_list[i] if length_control_bool else None
        knowledge_state = knowledge_states[i] if knowledge_states else None
        problem_id = problem_ids[i] if problem_ids else None
        data = {
            "problem": problem,
            "problem_id": problem_id,
            "user_profile": user_profiles[i],
            "length_control": length_control,
            "knowledge_state": knowledge_state,
            "knowledge_state_history": [knowledge_state] if knowledge_state else [],
            "conversation": [],
            "conversation_history": "",
            "assistant_messages": [],
            "first_query": True,
            "turns": 0,
            "finished": False,
            "over_max": False,
        }
        assistant_system_prompt = {
            "role": "system",
            "content": (
                "You are a skilled math tutor. Your goal is to help students understand and "
                "solve problems independently. Provide guidance based on their questions or "
                "mistakes. Ask questions to encourage their thinking and let students do most "
                "of the work themselves. Never give out the solution directly to students."
            ),
        }
        data["assistant_messages"].append(assistant_system_prompt)
        conversations_data.append(data)

    for turn in range(max_turns):
        user_full_contexts = []
        active_conversations = []
        for data in conversations_data:
            if data["finished"] or data["over_max"]:
                continue
            if data["first_query"]:
                if length_control_bool:
                    user_message_content = prompt_initial_query_template.format(
                        user_profile=data["user_profile"],
                        message_style=data["user_profile"],
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                        length_control=data["length_control"],
                        knowledge_state_formatted=_format_knowledge_state(data.get("knowledge_state")),
                        askable_concepts=_get_askable_concepts(data.get("knowledge_state")),
                        unknown_unknown_concepts=_get_unknown_unknown_concepts(data.get("knowledge_state")),
                        assistant_message=_get_last_assistant_message(data.get("assistant_messages", [])),
                    )
                else:
                    user_message_content = prompt_initial_query_template.format(
                        user_profile=data["user_profile"],
                        message_style=data["user_profile"],
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                        knowledge_state_formatted=_format_knowledge_state(data.get("knowledge_state")),
                        askable_concepts=_get_askable_concepts(data.get("knowledge_state")),
                        unknown_unknown_concepts=_get_unknown_unknown_concepts(data.get("knowledge_state")),
                        assistant_message=_get_last_assistant_message(data.get("assistant_messages", [])),
                    )
                data["first_query"] = False
            else:
                if length_control_bool:
                    user_message_content = prompt_template.format(
                        user_profile=data["user_profile"],
                        message_style=data["user_profile"],
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                        length_control=data["length_control"],
                        knowledge_state_formatted=_format_knowledge_state(data.get("knowledge_state")),
                        askable_concepts=_get_askable_concepts(data.get("knowledge_state")),
                        unknown_unknown_concepts=_get_unknown_unknown_concepts(data.get("knowledge_state")),
                        assistant_message=_get_last_assistant_message(data.get("assistant_messages", [])),
                        misguided_attempt_hint=_get_misguided_attempt_hint(
                            data.get("knowledge_state"),
                            data.get("knowledge_state_history", []),
                        ),
                    )
                else:
                    user_message_content = prompt_template.format(
                        user_profile=data["user_profile"],
                        message_style=data["user_profile"],
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                        knowledge_state_formatted=_format_knowledge_state(data.get("knowledge_state")),
                        askable_concepts=_get_askable_concepts(data.get("knowledge_state")),
                        unknown_unknown_concepts=_get_unknown_unknown_concepts(data.get("knowledge_state")),
                        assistant_message=_get_last_assistant_message(data.get("assistant_messages", [])),
                        misguided_attempt_hint=_get_misguided_attempt_hint(
                            data.get("knowledge_state"),
                            data.get("knowledge_state_history", []),
                        ),
                    )

            user_messages = [{"role": "user", "content": user_message_content}]
            data["user_messages"] = user_messages
            user_full_contexts.append(user_messages)
            active_conversations.append(data)

        if not active_conversations:
            break

        user_queries = await user_model_client.generate_responses(
            user_full_contexts,
            temperature=user_temperature,
            max_tokens=max_tokens,
            show_progress=show_progress,
        )

        for data, user_query in zip(active_conversations, user_queries):
            user_query_text = user_query[0] if user_query else ""
            if user_query_text.strip().lower() == "terminate: true":
                data["finished"] = True
                continue
            data["conversation"].append(("user", user_query_text))
            if "Thought:" in user_query_text:
                try:
                    if "Response:" in user_query_text:
                        query = user_query_text.split("Response:")[1].strip()
                    elif "Query:" in user_query_text:
                        query = user_query_text.split("Query:")[1].strip()
                    else:
                        query = user_query_text.split("Message:")[1].strip()
                except Exception:
                    data["finished"] = True
                    continue
            else:
                query = user_query_text

            if not user_query_text:
                data["finished"] = True
                continue

            if len(data["assistant_messages"]) == 1:
                first_turn_user_query = (
                    f"Here is the problem that you will tutor me on:\n"
                    f"{data['problem'].strip()}\n\n{query}"
                )
                data["assistant_messages"].append({"role": "user", "content": first_turn_user_query})
                data["first_query_content"] = query
            else:
                data["assistant_messages"].append({"role": "user", "content": query})

        active_conversations = [data for data in active_conversations if not data["finished"]]
        if not active_conversations:
            break

        assistant_full_contexts = [data["assistant_messages"] for data in active_conversations]
        assistant_responses = await assistant_model_client.generate_responses(
            assistant_full_contexts,
            temperature=assistant_temperature,
            max_tokens=max_tokens,
            show_progress=show_progress,
        )

        for data, assistant_response in zip(active_conversations, assistant_responses):
            assistant_text = assistant_response[0] if assistant_response else ""
            data["conversation"].append(("assistant", assistant_text))
            last_user_message = data["assistant_messages"][-1]["content"]
            if len(data["assistant_messages"]) == 2:
                last_user_message = data["first_query_content"]
            data["conversation_history"] += f"- You: {last_user_message}\n- AI Tutor: {assistant_text}\n"
            data["assistant_messages"].append({"role": "assistant", "content": assistant_text})
            data["turns"] += 1
            if not assistant_text:
                data["finished"] = True
            if data["turns"] >= max_turns:
                data["over_max"] = True

            if data.get("knowledge_state") is not None and concept_graph is not None and data.get("problem_id"):
                from ..knowledge.extract import extract_explained_concepts
                from ..knowledge.update import update_dynamic_knowledge_state

                concept_items = concept_graph.get(str(data["problem_id"]), [])
                concept_names = [item.get("concept_id", "") for item in concept_items if item.get("concept_id")]
                if concept_names:
                    explained_concepts = await extract_explained_concepts(
                        assistant_message=assistant_text,
                        candidate_concepts=concept_names,
                        model_client=user_model_client,
                        max_tokens=600,
                        show_progress=False,
                    )
                    data["explained_concepts_history"] = data.get("explained_concepts_history", [])
                    data["explained_concepts_history"].append(explained_concepts)
                    updated_state = await update_dynamic_knowledge_state(
                        assistant_message=assistant_text,
                        concept_names=explained_concepts or concept_names,
                        knowledge_state=data["knowledge_state"],
                        user_response_analysis=last_user_message,
                        model_client=user_model_client,
                        max_tokens=1200,
                        show_progress=False,
                        concept_graph=concept_graph,
                        problem_id=str(data.get("problem_id")) if data.get("problem_id") else None,
                    )
                    data["knowledge_state"] = updated_state
                    data["knowledge_state_history"].append(updated_state)

    return conversations_data

