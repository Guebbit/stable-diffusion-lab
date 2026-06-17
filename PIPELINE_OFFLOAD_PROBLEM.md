# PIPELINE_OFFLOAD — Problem Log

This document records all the problems discovered and fought during the implementation of
`PIPELINE_OFFLOAD` and `QUANTIZE_MODE`. The current state is a **working workaround** for
diffusers 0.30, but the underlying problems are not fully resolved. A future pass with more
time (or a more capable model) should revisit the open issues at the bottom.

---

## Background: Why This Matters

**FLUX.1-schnell** is the primary model driving these issues. Its architecture is unusual
compared to older SD models:

| Component       | Model       | VRAM (bfloat16) |
|-----------------|-------------|-----------------|
| Transformer     | FLUX        | ~12 GB          |
| Text encoder 1  | CLIP-L      | ~250 MB         |
| Text encoder 2  | **T5-XXL**  | **~9 GB**       |
| VAE             | FLUX        | ~400 MB         |
| **Total**       |             | **~22 GB**      |

The T5-XXL text encoder is the unusual part. SD 1.5 and SDXL use small CLIP encoders only.
FLUX adds T5-XXL, which is as large as many full language models. On an RTX 4090 (24 GB
VRAM) the entire model *barely* fits (22 GB / 24 GB). Every loading decision becomes
critical.

---

## PIPELINE_OFFLOAD — What Each Mode Does

Configured via `PIPELINE_OFFLOAD` env var. Implemented in
`backend/app/adapters/base.py` → `load_pipeline()`.

### `none` (recommended for 16 GB+ VRAM)
Load everything into VRAM at once. Fastest inference — all tensors already on GPU.
If the model doesn't fit, fails cleanly (or should; see the "balanced dilemma" below).

### `model_cpu` (default; recommended for 8–12 GB VRAM)
Keep the full model in CPU RAM. Register hooks that move each **sub-model** (text encoder,
transformer, VAE) to GPU only for its forward pass, then immediately back to CPU.
- Peak VRAM ≈ largest single sub-model (~12 GB for FLUX's transformer)
- RAM occupied permanently ≈ full model (~22 GB for FLUX)
- Inference is slower due to CPU↔GPU transfers between each sub-model

### `sequential_cpu` (recommended for 4–6 GB VRAM)
Register hooks that move each **individual layer** (not sub-model) to GPU for its forward
pass. ~1–2 GB peak VRAM regardless of model size. Slowest of all three.

---

## QUANTIZE_MODE — What Each Mode Does

Configured via `QUANTIZE_MODE` env var. Uses bitsandbytes NF4 4-bit quantization.
Implemented in `backend/app/adapters/base.py` → `build_quantized_kwargs()`.

**Important constraint:** bitsandbytes 4-bit tensors are CUDA-resident and cannot be moved
to CPU for offloading. So `QUANTIZE_MODE` + `model_cpu`/`sequential_cpu` is incompatible.
The code auto-overrides offload to `none` when quantization is set, with a logged warning.

### `none` (default)
No quantization. Full bfloat16 precision.

### `transformer`
Quantize the main generative model to NF4 4-bit:
- FLUX transformer: ~12 GB → ~3 GB
- SDXL UNet: ~4 GB → ~1 GB
- SD1.5 UNet: ~2 GB → ~500 MB (marginal benefit)

### `text_encoder`
Quantize T5-based text encoders to NF4 4-bit:
- FLUX T5-XXL (`text_encoder_2`): ~9 GB → ~2.5 GB
- SD3 T5-XXL (`text_encoder_3`): ~9 GB → ~2.5 GB
- **No effect on SD1.5/SDXL** — those use CLIP-only encoders (~250 MB, not worth quantizing)

### `full`
Both of the above. For FLUX: ~22 GB → ~5.5 GB total. Fits on 6–8 GB VRAM cards.

---

## Bug Log: Everything We Found and Fixed

### Bug 1 — Wrong default: `model_cpu` filling all RAM permanently

**Symptom:** User (RTX 4090, 32 GB RAM) reported FLUX filling all 22 GB of free RAM and
running inference from swap. GPU mostly idle.

**Root cause:** `settings.py` default is `pipeline_offload: str = "model_cpu"`. The user
hadn't set `PIPELINE_OFFLOAD` in their `.env`, so they silently got `model_cpu` mode.
This mode is **designed** to keep the full model (~22 GB for FLUX) in RAM permanently —
it's not a bug in `model_cpu`, it's the wrong mode for their hardware.

**Fix:** Documented the modes clearly in `.env-example` and `settings.py` comments.
Recommended `PIPELINE_OFFLOAD=none` for users with 16+ GB VRAM.

---

### Bug 2 — `float16` used instead of `bfloat16` for FLUX

**Symptom:** Poor image quality (washed out, artifacts) or black images with FLUX.

**Root cause:** All adapters used `dtype = torch.float16 if device == "cuda" else torch.float32`.
FLUX was trained in bfloat16. Its attention layers accumulate values that exceed float16's
maximum representable value (~65,504), causing NaN propagation through the network.

**Fix:** Changed to `dtype = torch.bfloat16 if device == "cuda" else torch.float32` in all
`_build_pipeline` methods (text_to_image, image_to_image, video, upscale, sketch_to_ink,
model_manager).

Note: bfloat16 requires Ampere or newer GPU (RTX 3000+ series). Older GPUs should be aware
of this change.

---

### Bug 3 — `sequential_cpu` used wrong mechanism (`device_map="sequential"`)

**Symptom:** `PIPELINE_OFFLOAD=sequential_cpu` didn't achieve the documented ~1–2 GB peak
VRAM. Actual VRAM usage was much higher.

**Root cause:** The original `text_to_image.py` and `image_to_image.py` used:
```python
pipeline = DiffusionPipeline.from_pretrained(
    model_path, device_map="sequential", ...
)
```

`device_map="sequential"` is an **accelerate distribution mode**: it fills GPU0, then
permanently spills remaining layers to GPU1 or CPU. It is not the same as
`enable_sequential_cpu_offload()` which installs **per-layer forward hooks** that temporarily
move each layer to GPU only for its forward pass. The VRAM profiles are completely different.

**Fix:** Changed to:
```python
pipeline = DiffusionPipeline.from_pretrained(model_path, ...)
pipeline.enable_sequential_cpu_offload()
```

---

### Bug 4 — `none` mode filled RAM via `device_map="balanced"` + free VRAM budget

**Symptom:** Even with `PIPELINE_OFFLOAD=none`, T5-XXL (and potentially other sub-components)
landed in CPU RAM instead of VRAM.

**Root cause (two parts):**

**Part A — free VRAM underestimates available budget:**
```python
free_bytes, _ = torch.cuda.mem_get_info(0)          # currently FREE VRAM (volatile!)
free_gb = max(free_bytes // (1024 ** 3) - 1, 1)    # underestimates real capacity
max_memory = {0: f"{free_gb}GiB", "cpu": "0GiB"}
```
On an RTX 4090 (24 GB total), if the display compositor and CUDA context claim 2 GB, then
`free_bytes` = 22 GB, `free_gb` = 21. FLUX at 22 GB > 21 GiB budget → accelerate overflows
1 GB to CPU. The `"cpu": "0GiB"` constraint was supposed to prevent this, but see Part B.

**Part B — `"cpu": "0GiB"` may not work as intended:**
It is unclear whether accelerate treats `"0GiB"` as "zero bytes allowed on CPU" or silently
ignores/misparses it. In practice, layers still ended up on CPU. This was not fully debugged
because the problem was superseded by Bug 5 (diffusers version constraint).

**Partial fix:** Changed the budget to use **total** VRAM capacity minus a fixed 2 GB headroom:
```python
total_vram = torch.cuda.get_device_properties(0).total_memory
budget_gb = max(total_vram // (1024 ** 3) - 2, 1)  # stable, ignores current usage
max_memory = {0: f"{budget_gb}GiB"}                 # no "cpu" key (see Bug 5 below)
```

For RTX 4090: budget = 22 GiB. FLUX at 22 GB ≤ 22 GiB → fits on GPU.

---

### Bug 5 — `device_map` dict rejected by diffusers 0.30 (current workaround)

**Symptom:** After attempting to use `device_map={"": 0}` (dict form that maps ALL
sub-components to cuda:0), the error `"device_map must be a string"` is raised.

**Root cause:** diffusers 0.30.0 has this validation in `pipeline_utils.py` (line 673):
```python
if device_map is not None and not isinstance(device_map, str):
    raise ValueError("`device_map` must be a string.")
```

And the only accepted string value is:
```python
SUPPORTED_DEVICE_MAP = ["balanced"]  # line 95
```

Note: The docstring for `from_pretrained` claims dict support (`Dict[str, Union[int, str, torch.device]]`)
but the code rejects it. This is a bug/regression in diffusers 0.30.

**Why the dict form was desirable:** `device_map={"": 0}` (the empty string `""` is a
catch-all matching every module name) forces ALL weights of ALL sub-components to go
directly from disk → cuda:0, with no CPU RAM staging at all. The `"balanced"` string
form uses accelerate's `infer_auto_device_map` which also avoids CPU staging, but has
more complex distribution logic and may not handle edge cases identically.

**Current workaround:** Use `device_map="balanced"` (the only valid string) with the
corrected `max_memory` budget (total VRAM - 2 GB), and add a post-load check:
```python
if hasattr(pipeline, "hf_device_map"):
    cpu_keys = [k for k, v in pipeline.hf_device_map.items() if str(v) in ("cpu", "disk")]
    if cpu_keys:
        raise InsufficientVRAMError(...)
```

**Remaining concern:** It is unconfirmed whether `device_map="balanced"` in diffusers 0.30
correctly covers **all** pipeline sub-components, specifically T5-XXL, which is loaded as
a separate HuggingFace Transformers model within the pipeline. If diffusers only applies
the device_map to its own diffusers-native components (transformer, VAE) and not to the
transformers-library components (T5EncoderModel, CLIPTextModel), T5-XXL could still land
on CPU despite the budget constraint. This is the **core unresolved question.**

---

## Current State (Summary)

| Mode | Status | Notes |
|------|--------|-------|
| `none` | ⚠️ Workaround | Uses `device_map="balanced"` + total VRAM budget. May still fail for T5-XXL coverage. Needs real-world testing. |
| `model_cpu` | ✅ Works | No device_map needed. Tested pattern. |
| `sequential_cpu` | ✅ Fixed | Now correctly uses `enable_sequential_cpu_offload()`. |
| `QUANTIZE_MODE` | ✅ Implemented | Untested — requires bitsandbytes installed. Architecture detection via model_index.json looks correct. |

---

## Open Problems for Future Investigation

### 1. Does `device_map="balanced"` actually cover T5-XXL in diffusers 0.30?

The critical question. Load FLUX with `PIPELINE_OFFLOAD=none` and check:
```python
print(pipeline.hf_device_map)  # should show all components on device 0, nothing on "cpu"
```
If T5-XXL shows `"cpu"`, the workaround doesn't fully work and a deeper fix is needed.

### 2. What diffusers version adds proper dict `device_map` support?

The dict form `{"": "cuda:0"}` or `{"": 0}` is documented in newer diffusers. Find the
version where it was (re-)introduced and note it in the docs. If the project can upgrade,
the workaround in Bug 5 can be replaced with the cleaner dict form.

### 3. Investigate `"cpu": "0GiB"` behaviour in accelerate

Does `max_memory={"cpu": "0GiB"}` actually prevent CPU usage in accelerate? Or does
accelerate silently ignore 0-valued entries? The answer affects whether we can force
hard GPU-only loading with a clean error when VRAM is insufficient.

### 4. Test `QUANTIZE_MODE` end-to-end

The quantization code in `build_quantized_kwargs()` (base.py) reads `model_index.json` to
detect architecture and pre-loads quantized sub-components. It has not been tested live.
Specifically verify:
- FLUX: `transformer` mode quantizes the transformer, `text_encoder` quantizes T5-XXL
- SDXL: `text_encoder` correctly skips the CLIP-based `text_encoder_2`
- Error handling when bitsandbytes is not installed

### 5. `model_cpu` VRAM peak comment is wrong for FLUX

The comment says "~3–5 GB peak VRAM" for `model_cpu`. For FLUX, peak VRAM during inference
is ~12 GB (FLUX transformer forward pass) not 3–5 GB. The 3–5 GB figure was accurate for
SD 1.5/SDXL UNets. Should be corrected once we have real measurements.

### 6. sketch_to_ink.py and upscale.py still use `apply_pipeline_to_device`

These adapters use a different loading path (`from_pretrained` → `apply_pipeline_to_device`)
that still does CPU staging for the `none` mode (via `.to(device)` after loading). For
small models (SD x4 upscaler ~800 MB, ControlNet ~1.5 GB) this is acceptable. But if a
user loads a large base model via sketch_to_ink (e.g. SDXL as base), they would hit the
CPU staging issue. Longer term: unify these adapters to also use `load_pipeline()`.

---

## Key Files

| File | Role |
|------|------|
| `backend/app/adapters/base.py` | `load_pipeline()`, `build_quantized_kwargs()`, `apply_pipeline_to_device()` |
| `backend/app/infrastructure/config/settings.py` | `pipeline_offload`, `quantize_mode` settings |
| `.env-example` | User-facing documentation of all modes |
| `backend/app/adapters/direct/text_to_image.py` | Main adapter using `load_pipeline()` |
| `backend/app/adapters/direct/image_to_image.py` | Same |
| `backend/app/adapters/direct/video.py` | Same |
| `backend/app/adapters/direct/model_manager.py` | Generic model loader using `load_pipeline()` |
| `backend/app/adapters/direct/upscale.py` | Still uses `apply_pipeline_to_device` (small model) |
| `backend/app/adapters/direct/sketch_to_ink.py` | Still uses `apply_pipeline_to_device` (complex multi-model) |
