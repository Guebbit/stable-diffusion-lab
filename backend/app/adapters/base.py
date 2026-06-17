"""
Base adapter utilities — shared logic for generation adapters.

Provides reusable helpers for progress tracking and artifact persistence
so each concrete adapter only describes its unique workflow steps.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any


def resolve_model_path(model_id: str) -> str:
    """
    Return the local filesystem path for a downloaded model, or model_id as-is.

    Download jobs store files at:
      settings.models_path / <source> / <model_id parts>
    e.g. /app/storage/models/huggingface/black-forest-labs/FLUX.1-dev

    Returning the local path lets from_pretrained() load entirely from disk,
    which avoids HuggingFace network calls — critical for gated models where
    a 401 would be raised even when files are already present locally.
    """
    from app.infrastructure.config.settings import get_settings
    models_path = get_settings().models_path
    parts = [p for p in model_id.replace("\\", "/").split("/") if p]
    for source in ("huggingface", "civitai", "github", "local"):
        candidate = models_path.joinpath(source, *parts)
        if candidate.is_dir() and any(candidate.iterdir()):
            return str(candidate)
    return model_id


def hf_token() -> str | None:
    """Return the HuggingFace API token from settings (None if not configured)."""
    from app.infrastructure.config.settings import get_settings
    token = get_settings().huggingface_token
    return token or None

from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference, GenerationParams, JobProgress


# Type alias for a raw step callback (current_step, total_steps)
StepCallback = Callable[[int, int], None]


class GenerationCancelledError(Exception):
    """Raised by the step callback when cooperative cancellation is requested."""


def build_diffusers_step_callback(
    on_progress: ProgressCallback | None,
    total_steps: int,
    cancel_event: Any = None,  # threading.Event | None — avoided circular import
) -> Callable[..., Any]:
    """
    Return a diffusers-compatible step callback: (pipe, step, timestep, kwargs) -> kwargs.

    Forwards JobProgress on each diffusion step. Raises GenerationCancelledError
    if cancel_event is set — this propagates out of the pipeline call cooperatively.
    Safe to call from a thread pool (the ProgressCallback is invoked synchronously).
    """

    def _callback(pipe: Any, step: int, timestep: Any, kwargs: Any) -> Any:
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Generation cancelled by user")
        if on_progress is None:
            return kwargs
        on_progress(
            JobProgress(
                job_id=uuid.UUID(int=0),
                status="running",
                progress_percent=int((step / max(total_steps, 1)) * 100),
                current_step=step,
                total_steps=total_steps,
            )
        )
        return kwargs

    return _callback


def build_legacy_diffusers_step_callback(
    on_progress: ProgressCallback | None,
    total_steps: int,
) -> Callable[..., None]:
    """
    Return an old-style diffusers step callback: (step, timestep, latents) -> None.

    Use this for pipelines that pre-date callback_on_step_end, such as
    StableDiffusionXLAdapterPipeline. Pass as callback=..., callback_steps=1.
    """

    def _callback(step: int, timestep: Any, latents: Any) -> None:
        if on_progress is None:
            return
        on_progress(
            JobProgress(
                job_id=uuid.UUID(int=0),
                status="running",
                progress_percent=int((step / max(total_steps, 1)) * 100),
                current_step=step,
                total_steps=total_steps,
            )
        )

    return _callback


def build_step_callback(
    on_progress: ProgressCallback | None,
    total_steps: int,
) -> StepCallback:
    """
    Return a (current, total) → None callback that forwards JobProgress.

    Wraps the user-provided ProgressCallback so adapters don't repeat
    the JobProgress construction boilerplate.
    """

    def _callback(current: int, total: int) -> None:
        if on_progress is None:
            return
        on_progress(
            JobProgress(
                job_id=uuid.UUID(int=0),
                status="running",
                progress_percent=int((current / max(total, 1)) * 100),
                current_step=current,
                total_steps=total,
            )
        )

    return _callback


class InsufficientVRAMError(RuntimeError):
    """Raised when free VRAM is too low to load a pipeline with PIPELINE_OFFLOAD=none."""


def estimate_pipeline_vram_mb(pipeline: Any) -> int:
    """
    Estimate VRAM usage in MB for a diffusers pipeline.

    Checks .transformer first (FLUX, DiT-based) then .unet (SD 1.x/2.x/XL).
    Returns 0 if neither attribute is found.
    """
    component = getattr(pipeline, "transformer", None) or getattr(pipeline, "unet", None)
    if component is None:
        return 0
    param_bytes = sum(p.numel() * p.element_size() for p in component.parameters())
    return (param_bytes * 2) // (1024 * 1024)


def apply_pipeline_to_device(pipeline: Any, device: str, offload_strategy: str) -> Any:
    """
    Move a pipeline (already in CPU RAM) to the target device using the offload strategy.

    Used by adapters that must load sub-components individually before assembling the
    pipeline (e.g. upscale.py loading StableDiffusionUpscalePipeline, sketch_to_ink.py
    injecting a ControlNet). Those adapters call from_pretrained() themselves first, then
    hand the result here for device placement.

    For standard text-to-image / img2img / video pipelines use load_pipeline() instead.
    load_pipeline() uses device_map={"": 0} in "none" mode — it streams weights directly
    from disk to VRAM without staging the full model through CPU RAM first. For FLUX
    (~22 GB), the staging done here would fill all available system RAM before a single
    tensor reaches the GPU.

    Strategies (set via PIPELINE_OFFLOAD env var):
      none           — move entire pipeline to VRAM; raises InsufficientVRAMError if it won't fit.
      model_cpu      — stream sub-models to GPU one at a time (peak VRAM ≈ largest sub-model).
      sequential_cpu — stream layer-by-layer (~1-2 GB peak VRAM, slowest).
    """
    import logging
    logger = logging.getLogger(__name__)

    if device != "cuda":
        return pipeline.to(device)

    if offload_strategy == "none":
        import torch
        needed_mb = estimate_pipeline_vram_mb(pipeline)
        if needed_mb > 0:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_mb = free_bytes // (1024 * 1024)
            total_mb = total_bytes // (1024 * 1024)
            if free_mb < needed_mb:
                raise InsufficientVRAMError(
                    f"Not enough VRAM to load pipeline: need ~{needed_mb} MB, "
                    f"only {free_mb} MB free of {total_mb} MB total. "
                    "Free GPU memory (stop other jobs or models) or set "
                    "PIPELINE_OFFLOAD=model_cpu in your .env."
                )
            logger.info(
                "VRAM check passed: need ~%d MB, %d MB free — loading to GPU", needed_mb, free_mb
            )
        return pipeline.to(device)

    if offload_strategy == "sequential_cpu":
        pipeline.enable_sequential_cpu_offload()
        return pipeline

    # Default: model_cpu
    pipeline.enable_model_cpu_offload()
    return pipeline


def load_pipeline(
    pipeline_class: type,
    model_path: str,
    device: str,
    dtype: Any,
    offload_strategy: str,
    quantize_mode: str,
    token: str | None,
    **extra_kwargs: Any,
) -> Any:
    """
    Unified pipeline loader — handles all combinations of offload strategy and quantization.

    Replaces the per-adapter loading blocks that previously duplicated and diverged from
    each other (text_to_image.py, image_to_image.py, video.py had three separate copies).
    Fixes three bugs found during FLUX.1 debugging:

    Bug 1 — device_map="balanced" filled system RAM with large models:
        The original "none" mode used device_map="balanced" which asks accelerate to
        distribute layers across available devices. For multi-component pipelines (FLUX,
        SDXL), sub-components like T5-XXL are loaded independently by diffusers and were
        never covered by the device_map — they always ended up in CPU RAM regardless of
        the "cpu": "0GiB" max_memory constraint. For FLUX (~22 GB), this silently filled
        all free system RAM and triggered swap, turning a fast GPU into a swap-bound crawl.
        Fix: device_map={"": 0} maps EVERY parameter of EVERY sub-component to cuda:0.
        The empty-string prefix is a catch-all that matches any module name. Weights stream
        directly from disk → VRAM with minimal CPU staging.

    Bug 2 — float16 caused NaN in FLUX attention layers:
        FLUX.1 was trained in bfloat16. Its attention layers accumulate intermediate
        values that exceed float16's max (~65504), causing NaN propagation that produced
        black images or very low-quality output. The dtype fix lives in each adapter
        (bfloat16 for CUDA), not in this function, but is documented here for context.

    Bug 3 — sequential_cpu used device_map="sequential" (wrong mechanism):
        device_map="sequential" is an accelerate distribution mode: fill GPU0, then spill
        remaining layers permanently to GPU1/CPU. enable_sequential_cpu_offload() is an
        entirely different mechanism: it installs per-layer PyTorch forward hooks that
        temporarily move each layer to GPU only for its forward pass (~1-2 GB peak VRAM).
        The two have completely different VRAM profiles; the old code gave SDXL-level VRAM
        usage instead of the intended 1-2 GB.

    Args:
        pipeline_class:   Diffusers pipeline class (DiffusionPipeline, AutoPipelineForImage2Image…)
        model_path:       Local filesystem path or HuggingFace model ID.
        device:           "cuda" or "cpu".
        dtype:            torch.bfloat16 (CUDA) or torch.float32 (CPU).
        offload_strategy: "none", "model_cpu", or "sequential_cpu" (from PIPELINE_OFFLOAD).
        quantize_mode:    "none", "transformer", "text_encoder", or "full" (from QUANTIZE_MODE).
        token:            HuggingFace API token, or None.
        **extra_kwargs:   Forwarded to pipeline_class.from_pretrained()
                          (e.g. safety_checker=None, custom VAE, LoRA weights).
    """
    import logging
    logger = logging.getLogger(__name__)

    # CPU path: bitsandbytes requires CUDA, and offloading is pointless on CPU.
    if device != "cuda":
        if quantize_mode != "none":
            logger.warning(
                "QUANTIZE_MODE=%r is ignored on CPU — bitsandbytes requires CUDA.",
                quantize_mode,
            )
        return pipeline_class.from_pretrained(
            model_path, torch_dtype=dtype, token=token, **extra_kwargs
        )

    # bitsandbytes 4-bit tensors are CUDA-resident; they cannot be moved to CPU for
    # offloading. If the user set both quantize_mode and a CPU offload strategy, override
    # offload to "none" so the quantized model stays on GPU where bitsandbytes needs it.
    effective_offload = offload_strategy
    if quantize_mode != "none" and offload_strategy != "none":
        logger.warning(
            "QUANTIZE_MODE=%r overrides PIPELINE_OFFLOAD=%r — bitsandbytes 4-bit tensors "
            "are CUDA-resident and cannot be CPU-offloaded. Using PIPELINE_OFFLOAD=none.",
            quantize_mode,
            offload_strategy,
        )
        effective_offload = "none"

    # Pre-load quantized sub-components for injection into from_pretrained.
    # Must happen BEFORE the main from_pretrained call because bitsandbytes
    # quantization occurs during weight loading, not as a post-hoc step.
    quantized_kwargs = build_quantized_kwargs(model_path, quantize_mode, dtype, token)

    # Caller's extra_kwargs take precedence (e.g. a pre-loaded LoRA or custom VAE
    # should not be overwritten by a freshly-quantized version of the same component).
    load_kwargs = {**quantized_kwargs, **extra_kwargs}

    if effective_offload == "none":
        # Load all sub-models directly to GPU via accelerate's balanced device_map.
        #
        # Why "balanced" and not the dict form {"": 0}?
        # diffusers 0.30.x validates `isinstance(device_map, str)` and only accepts
        # the string "balanced" (SUPPORTED_DEVICE_MAP = ["balanced"]). Dict device_map
        # is documented but rejected at runtime in this version.
        #
        # Why total VRAM instead of torch.cuda.mem_get_info() (free VRAM)?
        # Free VRAM is volatile — the desktop compositor and CUDA context already claim
        # 1-3 GB. Using free VRAM as the budget underestimates what is actually
        # available for the model, causing accelerate to spill layers to CPU even when
        # the card is fully capable of holding them. Using total VRAM capacity (minus a
        # fixed 2 GB headroom for activations) correctly reflects the card's budget.
        # Example: RTX 4090 (24 GB), 2 GB used by system:
        #   old: free=22 GB → budget=21 GiB → FLUX 22 GB OVERFLOWS to CPU ✗
        #   new: total=24 GB → budget=22 GiB → FLUX 22 GB fits on GPU ✓
        #
        # Omitting "cpu" from max_memory lets accelerate use CPU as a last-resort
        # overflow rather than raising a hard error — we detect the overflow below
        # and raise InsufficientVRAMError ourselves with a clear remediation message.
        import torch
        total_vram = torch.cuda.get_device_properties(0).total_memory
        budget_gb = max(total_vram // (1024 ** 3) - 2, 1)
        logger.info(
            "Loading pipeline %s to GPU via balanced device_map "
            "(budget=%d GiB of %d GiB total, PIPELINE_OFFLOAD=none)",
            model_path, budget_gb, total_vram // (1024 ** 3),
        )
        pipeline = pipeline_class.from_pretrained(
            model_path,
            torch_dtype=dtype,
            token=token,
            device_map="balanced",
            max_memory={0: f"{budget_gb}GiB"},
            **load_kwargs,
        )
        # If anything overflowed to CPU, VRAM is genuinely insufficient for none mode.
        # hf_device_map is set by accelerate whenever device_map is used.
        if hasattr(pipeline, "hf_device_map"):
            cpu_keys = [
                k for k, v in pipeline.hf_device_map.items()
                if str(v) in ("cpu", "disk")
            ]
            if cpu_keys:
                raise InsufficientVRAMError(
                    f"PIPELINE_OFFLOAD=none: not enough VRAM — "
                    f"{len(cpu_keys)} component(s) landed on CPU "
                    f"(e.g. {cpu_keys[0]!r}). "
                    "Options: free GPU memory, set QUANTIZE_MODE=full, or switch to "
                    "PIPELINE_OFFLOAD=model_cpu."
                )
        return pipeline

    if effective_offload == "model_cpu":
        # Load to CPU RAM first, then install sub-model CPU offload hooks.
        # enable_model_cpu_offload() keeps the full model in CPU RAM and registers
        # hooks that move each whole sub-model (text_encoder, transformer, vae…) to
        # GPU only for its forward pass, then immediately back to CPU.
        # Peak VRAM ≈ largest single sub-model (e.g. ~12 GB for FLUX transformer).
        # Trade-off: the full model (~22 GB for FLUX) occupies CPU RAM at all times.
        # Best for: GPUs with 8–12 GB VRAM + sufficient system RAM.
        logger.info(
            "Loading pipeline %s to CPU RAM for sub-model offload (PIPELINE_OFFLOAD=model_cpu)",
            model_path,
        )
        pipeline = pipeline_class.from_pretrained(
            model_path,
            torch_dtype=dtype,
            token=token,
            **load_kwargs,
        )
        pipeline.enable_model_cpu_offload()
        return pipeline

    if effective_offload == "sequential_cpu":
        # Load to CPU RAM first, then install per-layer sequential offload hooks.
        # enable_sequential_cpu_offload() hooks every individual nn.Module so each
        # layer moves to GPU only for its own forward pass (~1-2 GB peak VRAM).
        # This is the correct mechanism — NOT device_map="sequential" which is an
        # accelerate layer-distribution mode with a completely different VRAM profile.
        # Best for: GPUs with 4–6 GB VRAM where model_cpu is still too large.
        logger.info(
            "Loading pipeline %s to CPU RAM for sequential layer offload "
            "(PIPELINE_OFFLOAD=sequential_cpu)",
            model_path,
        )
        pipeline = pipeline_class.from_pretrained(
            model_path,
            torch_dtype=dtype,
            token=token,
            **load_kwargs,
        )
        pipeline.enable_sequential_cpu_offload()
        return pipeline

    # Unknown strategy — fall back to the "none" behaviour and warn.
    logger.warning(
        "Unknown PIPELINE_OFFLOAD=%r — falling back to direct GPU load (none behaviour).",
        offload_strategy,
    )
    import torch
    total_vram = torch.cuda.get_device_properties(0).total_memory
    budget_gb = max(total_vram // (1024 ** 3) - 2, 1)
    return pipeline_class.from_pretrained(
        model_path,
        torch_dtype=dtype,
        token=token,
        device_map="balanced",
        max_memory={0: f"{budget_gb}GiB"},
        **load_kwargs,
    )


def build_quantized_kwargs(
    model_path: str,
    quantize_mode: str,
    dtype: Any,
    token: str | None,
) -> dict[str, Any]:
    """
    Pre-load quantized sub-components and return them for injection into from_pretrained.

    Returns a dict of {component_name: pre_loaded_component} to be unpacked as **kwargs
    when calling pipeline_class.from_pretrained(). Components NOT in the returned dict
    are loaded at full precision by diffusers in the normal way.

    Why inject pre-loaded components instead of passing quantization_config to from_pretrained?
    Diffusers' pipeline-level quantization_config (when it works at all) targets only the
    primary model component. FLUX has two heavy components: the transformer (12 GB) and the
    T5-XXL text encoder (9 GB). Loading them individually gives explicit control over which
    are quantized and lets us leave the small CLIP encoder (<250 MB) at full precision.

    Why NF4 instead of INT8?
    NF4 (Normal Float 4-bit, from the QLoRA paper) spaces its quantization bins to minimise
    rounding error for normally-distributed values — which neural network weights almost
    universally are. INT4/INT8 use uniformly-spaced bins that waste resolution on the tails.
    At 4-bit NF4, perceptual quality loss is minimal (~0.5–1 dB PSNR for most diffusion models).

    Modes:
      none         → {} (early return; bitsandbytes never imported)
      transformer  → {"transformer": quantized} or {"unet": quantized}
      text_encoder → {"text_encoder_2": quantized} and/or {"text_encoder_3": quantized}
                     only if those slots hold T5-family models (FLUX, SD3).
                     SDXL's text_encoder_2 is CLIPTextModelWithProjection — skipped.
      full         → transformer/unet + T5 encoder(s)
    """
    if quantize_mode == "none":
        return {}

    import logging
    logger = logging.getLogger(__name__)

    try:
        import bitsandbytes as _bnb  # noqa: F401
    except ImportError:
        raise RuntimeError(
            f"QUANTIZE_MODE={quantize_mode!r} requires the bitsandbytes package. "
            "Install it with:  pip install bitsandbytes"
        ) from None

    from diffusers import DiffusionPipeline
    from transformers import BitsAndBytesConfig

    # NF4 4-bit configuration.
    # load_in_4bit=True         — store weights as 4-bit NF4 (quarter the float16 footprint)
    # bnb_4bit_quant_type="nf4" — use NF4 grid instead of INT4 (better quality per bit)
    # bnb_4bit_compute_dtype    — dequantise to this dtype during the actual forward pass
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )

    # Read model_index.json to discover what sub-components this pipeline has without
    # hard-coding model family names. Works for FLUX, SDXL, SD3, or any future arch.
    pipeline_config = DiffusionPipeline.load_config(model_path)
    logger.debug("Pipeline config for quantization: %s", list(pipeline_config.keys()))

    quantized: dict[str, Any] = {}

    if quantize_mode in ("transformer", "full"):
        quantized.update(
            _load_quantized_main_model(model_path, pipeline_config, nf4_config, dtype, token)
        )

    if quantize_mode in ("text_encoder", "full"):
        quantized.update(
            _load_quantized_text_encoders(model_path, pipeline_config, nf4_config, dtype, token)
        )

    return quantized


def _load_quantized_main_model(
    model_path: str,
    pipeline_config: dict,
    nf4_config: Any,
    dtype: Any,
    token: str | None,
) -> dict[str, Any]:
    """
    Quantize the main generative model to NF4 4-bit.

    Detects architecture from pipeline_config (model_index.json):
      - "transformer" key → FLUX, SD3, or any DiT-based model
      - "unet" key       → SD 1.x, SD 2.x, SDXL

    Dynamically imports the component class instead of hard-coding model families,
    so this works for future architectures too.

    VRAM savings (NF4 4-bit vs bfloat16):
      FLUX transformer:   ~12 GB → ~3 GB
      SDXL UNet:           ~4 GB → ~1 GB
      SD1.5 UNet:          ~2 GB → ~500 MB  (already small, marginal benefit)
    """
    import importlib
    import logging
    logger = logging.getLogger(__name__)

    # Transformer-based models (FLUX, SD3, DiT variants)
    if "transformer" in pipeline_config:
        module_name, class_name = pipeline_config["transformer"]
        logger.info(
            "QUANTIZE_MODE: loading %s in NF4 4-bit from %s/transformer",
            class_name, model_path,
        )
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
            component = cls.from_pretrained(
                model_path,
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=dtype,
                token=token,
            )
            return {"transformer": component}
        except Exception as exc:
            logger.warning(
                "Could not quantize %s — falling back to full precision: %s",
                class_name, exc,
            )
            return {}

    # UNet-based models (SD 1.x, SD 2.x, SDXL)
    if "unet" in pipeline_config:
        module_name, class_name = pipeline_config["unet"]
        logger.info(
            "QUANTIZE_MODE: loading %s in NF4 4-bit from %s/unet",
            class_name, model_path,
        )
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
            component = cls.from_pretrained(
                model_path,
                subfolder="unet",
                quantization_config=nf4_config,
                torch_dtype=dtype,
                token=token,
            )
            return {"unet": component}
        except Exception as exc:
            logger.warning(
                "Could not quantize %s — falling back to full precision: %s",
                class_name, exc,
            )
            return {}

    logger.warning(
        "QUANTIZE_MODE=transformer: no 'transformer' or 'unet' found in %s — skipped.",
        model_path,
    )
    return {}


def _load_quantized_text_encoders(
    model_path: str,
    pipeline_config: dict,
    nf4_config: Any,
    dtype: Any,
    token: str | None,
) -> dict[str, Any]:
    """
    Quantize T5-based text encoder sub-components to NF4 4-bit.

    Checks text_encoder_2 (FLUX pattern) and text_encoder_3 (SD3 pattern).
    Skips CLIP-based encoders (CLIPTextModel, CLIPTextModelWithProjection) because
    they are ~250 MB — too small to bother quantizing.

    SDXL note: text_encoder_2 is CLIPTextModelWithProjection, not T5 — this function
    correctly skips it. QUANTIZE_MODE=text_encoder has no effect on SD1.x / SDXL.

    VRAM savings:
      FLUX T5-XXL (text_encoder_2):  ~9 GB → ~2.5 GB
      SD3 T5-XXL  (text_encoder_3):  ~9 GB → ~2.5 GB
    """
    import importlib
    import logging
    logger = logging.getLogger(__name__)

    quantized: dict[str, Any] = {}

    # Both FLUX (text_encoder_2) and SD3 (text_encoder_3) may have a T5 encoder.
    for slot in ("text_encoder_2", "text_encoder_3"):
        if slot not in pipeline_config:
            continue

        module_name, class_name = pipeline_config[slot]

        # Only quantize T5 encoders — CLIP encoders are already small enough to skip.
        # "T5" appears in T5EncoderModel, T5ForConditionalGeneration, etc.
        if "T5" not in class_name:
            logger.debug(
                "Skipping %s quantization: %s is CLIP-based (~250 MB, not worth quantizing)",
                slot, class_name,
            )
            continue

        logger.info(
            "QUANTIZE_MODE: loading %s/%s in NF4 4-bit from %s",
            slot, class_name, model_path,
        )
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
            component = cls.from_pretrained(
                model_path,
                subfolder=slot,
                quantization_config=nf4_config,
                torch_dtype=dtype,
                token=token,
            )
            quantized[slot] = component
        except Exception as exc:
            logger.warning(
                "Could not quantize %s (%s) — falling back to full precision: %s",
                slot, class_name, exc,
            )

    if not quantized:
        logger.info(
            "QUANTIZE_MODE=text_encoder: no T5 encoders found in %s "
            "(SD1.x / SDXL use CLIP-only encoders — no quantization applied).",
            model_path,
        )

    return quantized


def save_artifacts_from_bytes(
    image_bytes_list: list[bytes],
    output_dir: Path,
    params: GenerationParams,
) -> list[ArtifactReference]:
    """
    Save raw image bytes to disk and return ArtifactReference list.

    Each image is saved as a unique PNG file. The caller is responsible
    for fetching the bytes (network download, pipeline output, etc.).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[ArtifactReference] = []

    for image_bytes in image_bytes_list:
        artifact_id = uuid.uuid4()
        local_filename = f"{artifact_id}.png"
        file_path = output_dir / local_filename
        file_path.write_bytes(image_bytes)

        artifacts.append(
            ArtifactReference(
                artifact_id=artifact_id,
                job_id=uuid.UUID(int=0),
                file_path=str(file_path),
                media_type="image/png",
                width=params.width,
                height=params.height,
                size_bytes=len(image_bytes),
            )
        )

    return artifacts


def save_artifacts_from_pil_images(
    images: list[Any],
    output_dir: Path,
    params: GenerationParams,
) -> list[ArtifactReference]:
    """
    Save PIL Image objects to disk and return ArtifactReference list.

    Used by Direct adapters after diffusers pipeline returns PIL images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[ArtifactReference] = []

    for image in images:
        artifact_id = uuid.uuid4()
        local_filename = f"{artifact_id}.png"
        file_path = output_dir / local_filename
        image.save(file_path)

        artifacts.append(
            ArtifactReference(
                artifact_id=artifact_id,
                job_id=uuid.UUID(int=0),
                file_path=str(file_path),
                media_type="image/png",
                width=params.width,
                height=params.height,
                size_bytes=file_path.stat().st_size,
            )
        )

    return artifacts
