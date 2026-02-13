"""Single-model adapter (ported from utils.generate_responses_in_batch)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import aiolimiter
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from .logging import build_log_entry, log_llm_calls, print_llm_calls


class SingleModelClient:
    """Adapter for a single configured model (no routing)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def _throttled_openai_chat_completion(
        self,
        client: AsyncOpenAI,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        n: int,
        limiter: aiolimiter.AsyncLimiter,
        reasoning_effort: Optional[str] = None,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        async with limiter:
            for _ in range(20):
                try:
                    params: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_completion_tokens": max_tokens,
                        "top_p": top_p,
                        "n": n,
                    }
                    if reasoning_effort is not None:
                        params["reasoning_effort"] = reasoning_effort
                    if json_mode:
                        params["response_format"] = {"type": "json_object"}
                    return await client.chat.completions.create(**params)
                except Exception as e:
                    raise e
        return {"choices": [{"message": {"content": ""}} for _ in range(n)]}

    def _map_model(self, model_name: str) -> str:
        if model_name == "gpt-4o":
            return "gpt-4o-2024-05-13"
        if model_name == "gpt-4o-241120":
            return "gpt-4o-2024-11-20"
        if model_name in ["gpt-5", "gpt-5-thinking"]:
            return "gpt-5-2025-08-07"
        if model_name in ["gpt-5-mini", "gpt-5-mini-thinking"]:
            return "gpt-5-mini-2025-08-07"
        if model_name in ["gpt-5-nano", "gpt-5-nano-thinking"]:
            return "gpt-5-nano-2025-08-07"
        return model_name

    async def generate_responses(
        self,
        full_contexts: List[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        n: int = 1,
        show_progress: bool = True,
        json_mode: bool = False,
    ) -> List[List[str]]:
        """
        Generate responses for a batch of contexts using a single OpenAI model.
        Returns a list of response lists (one list per context).
        """
        client = AsyncOpenAI()

        reasoning_effort = None
        if self.model_name in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            reasoning_effort = "minimal"
            temperature = 1.0
        elif self.model_name in ["gpt-5-thinking", "gpt-5-mini-thinking", "gpt-5-nano-thinking"]:
            reasoning_effort = "medium"
            temperature = 1.0

        actual_model = self._map_model(self.model_name)
        limiter = aiolimiter.AsyncLimiter(100, time_period=60)
        semaphore = asyncio.Semaphore(100)

        async def limited_task(context):
            async with semaphore:
                return await self._throttled_openai_chat_completion(
                    client=client,
                    model=actual_model,
                    messages=context,
                    temperature=temperature if temperature is not None else 0,
                    max_tokens=max_tokens,
                    top_p=1.0,
                    n=n,
                    limiter=limiter,
                    reasoning_effort=reasoning_effort,
                    json_mode=json_mode,
                )

        async_responses = [limited_task(context) for context in full_contexts]
        if show_progress:
            responses = await tqdm_asyncio.gather(*async_responses)
        else:
            responses = await asyncio.gather(*async_responses)

        generated_responses: List[List[str]] = []
        for resp in responses:
            scenario_responses: List[str] = []
            for i in range(n):
                try:
                    content = resp.choices[i].message.content
                    content = content.strip()
                except Exception:
                    content = ""
                scenario_responses.append(content)
            generated_responses.append(scenario_responses)

        await log_batch_calls(
            model_name=self.model_name,
            full_contexts=full_contexts,
            outputs=generated_responses,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
        return generated_responses


async def log_batch_calls(
    *,
    model_name: str,
    full_contexts: List[List[Dict[str, str]]],
    outputs: List[List[str]],
    temperature: float,
    max_tokens: int,
    n: int,
) -> None:
    log_entries = []
    for context, out in zip(full_contexts, outputs):
        system_prompt = ""
        user_prompt = ""
        for msg in context:
            if msg.get("role") == "system" and not system_prompt:
                system_prompt = msg.get("content", "")
        for msg in reversed(context):
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break
        log_entries.append(
            build_log_entry(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=context,
                output=out,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
        )
    await log_llm_calls(log_entries)
    print_llm_calls(log_entries)

