"""
BentoML LLM adapter — delegates chat completions to a BentoML runner.

Implements LLMProvider protocol. Sends chat messages, receives completion text.
"""

from __future__ import annotations

import logging

from app.adapters.bentoml.client import BentoMLClient


logger = logging.getLogger(__name__)


class BentoMLLLMAdapter:
    """Local LLM inference via BentoML service."""

    def __init__(self, client: BentoMLClient) -> None:
        self._client = client

    async def generate(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Request LLM chat completion from BentoML service.

        Sends the message history and generation params, receives the completion.
        """
        payload = {
            "model_id": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = await self._client.post_json(
            "/api/v1/generate/llm/chat",
            payload=payload,
            timeout_key="llm",
        )

        return response.get("content", "")
