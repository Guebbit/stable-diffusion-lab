"""
Direct LLM adapter — runs local language model inference.

Supports two sub-strategies:
1. HuggingFace transformers (AutoModelForCausalLM) for safetensors models
2. llama-cpp-python for GGUF quantized models

Auto-selects the correct strategy based on model file format.
Implements LLMProvider protocol from app.domain.protocols.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.adapters.direct.pipeline_cache import PipelineCache


logger = logging.getLogger(__name__)


class DirectLLMAdapter:
    """
    Local LLM inference using transformers or llama.cpp.

    Strategy selection:
    - If model_id points to a .gguf file → uses llama-cpp-python
    - Otherwise → uses transformers AutoModelForCausalLM

    Chat template formatting is handled internally — the adapter applies
    the tokenizer's chat template or falls back to standard formats.
    """

    def __init__(self, pipeline_cache: PipelineCache) -> None:
        self._cache = pipeline_cache

    async def generate(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a text completion from the message history.

        Loads the LLM model via PipelineCache, formats the chat template,
        runs inference, and returns the generated text.

        Args:
            messages: Chat history as [{"role": "user", "content": "..."}].
            model_id: HuggingFace model ID or path to GGUF file.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            Generated completion text.
        """
        # Determine backend strategy based on model format
        is_gguf = model_id.endswith(".gguf") or "/gguf/" in model_id.lower()

        if is_gguf:
            return await self._generate_gguf(messages, model_id, max_tokens, temperature)

        return await self._generate_transformers(messages, model_id, max_tokens, temperature)

    async def _generate_transformers(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using HuggingFace transformers (full/quantized models)."""
        # Get model + tokenizer from cache
        components = await self._cache.get_or_load(
            model_id,
            loader=lambda: asyncio.to_thread(self._build_transformers_pipeline, model_id),
            category="llm",
        )

        # Run inference on thread pool
        result = await asyncio.to_thread(
            self._run_transformers_inference,
            components,
            messages,
            max_tokens,
            temperature,
        )
        return result

    async def _generate_gguf(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using llama-cpp-python (GGUF quantized models)."""
        # Get llama.cpp model from cache
        llm = await self._cache.get_or_load(
            model_id,
            loader=lambda: asyncio.to_thread(self._build_gguf_model, model_id),
            category="llm",
        )

        # Run inference on thread pool
        result = await asyncio.to_thread(
            self._run_gguf_inference, llm, messages, max_tokens, temperature
        )
        return result

    @staticmethod
    def _build_transformers_pipeline(model_id: str) -> tuple[Any, int]:
        """
        Build transformers model + tokenizer — called by PipelineCache on miss.

        Attempts 4-bit quantization (BitsAndBytes) if available, falls back
        to float16 on GPU or float32 on CPU.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            from app.infrastructure.config.settings import get_settings
            if not get_settings().allow_cpu_fallback:
                raise RuntimeError(
                    "CUDA not available — LLM job aborted. "
                    "Set ALLOW_CPU_FALLBACK=true to allow CPU inference."
                )
            logger.warning("CUDA not available — running LLM on CPU (ALLOW_CPU_FALLBACK=true)")
        logger.info("Building LLM pipeline (transformers): %s → %s", model_id, device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Try quantized loading on GPU for memory efficiency
        model_kwargs: dict[str, Any] = {}
        if device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
                logger.info("Using 4-bit quantization for %s", model_id)
            except ImportError:
                # BitsAndBytes not installed — use float16
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        # Estimate VRAM
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        estimated_vram_mb = param_bytes // (1024 * 1024)

        components = {"model": model, "tokenizer": tokenizer, "device": device}
        return components, estimated_vram_mb

    @staticmethod
    def _build_gguf_model(model_id: str) -> tuple[Any, int]:
        """
        Build llama.cpp model from GGUF file — called by PipelineCache on miss.

        Loads the model with GPU layers offloaded for hybrid CPU/GPU inference.
        """
        from llama_cpp import Llama

        logger.info("Building LLM pipeline (llama.cpp): %s", model_id)

        # n_gpu_layers=-1 offloads all layers to GPU if available
        llm = Llama(
            model_path=model_id,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )

        # Rough VRAM estimate based on file size (GGUF models are pre-quantized)
        from pathlib import Path

        model_path = Path(model_id)
        size_mb = model_path.stat().st_size // (1024 * 1024) if model_path.exists() else 4000

        return llm, size_mb

    @staticmethod
    def _run_transformers_inference(
        components: dict[str, Any],
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Synchronous transformers inference — runs on thread pool."""
        import torch

        model = components["model"]
        tokenizer = components["tokenizer"]

        # Apply chat template to format messages into model-expected format
        if hasattr(tokenizer, "apply_chat_template"):
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation for models without chat template
            input_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
            input_text += "\nassistant:"

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        # Generate completion
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),  # Avoid division by zero
                do_sample=temperature > 0.01,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (not the input)
        generated_ids = output_ids[0][input_ids.shape[1] :]
        result = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return result.strip()

    @staticmethod
    def _run_gguf_inference(
        llm: Any,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Synchronous llama.cpp inference — runs on thread pool."""
        # llama-cpp-python has a built-in chat completion API
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract the assistant's response
        return response["choices"][0]["message"]["content"].strip()
