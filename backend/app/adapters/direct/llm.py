"""
Direct LLM adapter — runs local language model inference.

Uses transformers AutoModelForCausalLM for local LLM chat completions.
Implements LLMProvider protocol from app.domain.protocols.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class DirectLLMAdapter:
    """Local LLM inference using transformers causal language models."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a text completion from the message history.

        Loads the LLM model (if not cached), formats the chat template,
        and runs inference to produce a response.
        """
        # TODO: Implement LLM loading and inference
        raise NotImplementedError("DirectLLMAdapter.generate is not yet implemented")
