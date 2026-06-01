"""
ComfyUI LLM adapter — protocol-complete stub for LLM via ComfyUI.

ComfyUI is not typically used for LLM inference, but this adapter exists
to satisfy the protocol completeness requirement. It will raise a clear
error if called, directing users to the direct or BentoML backends.
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


class ComfyUILLMAdapter:
    """
    LLM inference via ComfyUI — protocol-complete stub.

    ComfyUI does not natively support LLM chat completions.
    This adapter raises NotImplementedError with a helpful message.
    """

    async def generate(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """LLM is not supported via ComfyUI backend."""
        raise NotImplementedError(
            "LLM inference is not available via ComfyUI backend. "
            "Use InferenceBackend.DIRECT_PYTHON or InferenceBackend.BENTOML instead."
        )
