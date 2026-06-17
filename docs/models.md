# Model Types Guide

A practical reference for the model types used in this project. Every model in the database has a `model_type` field that maps to one of these categories.

---

## The mental model: a stack

Most generation tasks involve more than one model. Think of it as a stack:

```
[ LoRA / IP-Adapter / ControlNet ]   ← optional helpers that modify or guide
[ VAE ]                               ← encodes/decodes pixel space (often implicit)
[ Base model ]                        ← the core generator — always required
```

You always need a **base model**. Everything else is optional and must be **compatible** with that base.

---

## Base models (`model_type: base_diffusion`)

The core generator. Takes a text prompt (and optionally an input image) and produces an output image.

**Families and generations** — base models come in generations that are NOT interchangeable. A LoRA or ControlNet trained for SDXL will not work on SD 1.5, and vice versa. The `base_model` field (e.g. `sdxl`, `sd1.5`, `flux`) encodes this.

| Family | `base_model` value | Typical resolution | Notes |
|---|---|---|---|
| Stable Diffusion 1.x | `sd1.5` | 512×512 | Oldest; lightest; widest ecosystem of helpers |
| Stable Diffusion 2.x | `sd2.1` | 768×768 | Improved quality but fewer community helpers |
| SDXL | `sdxl` | 1024×1024 | Large step up in quality; needs 8+ GB VRAM |
| FLUX | `flux` | 1024×1024 | Current flagship; transformer (DiT) architecture, no UNet; 12+ GB VRAM |

**Custom checkpoints** (e.g. RealVisXL, NovelAI XL) are fine-tuned versions of a base family. They share the same helpers as their parent family and have `family: custom` but `base_model: sdxl`.

**Negative prompts** — work well on SD 1.5/2/SDXL; have minimal effect on FLUX due to its architecture.

---

## VAE (`model_type: vae`)

The **Variational Autoencoder** translates between pixel space and latent space. The base model works in compressed latent space; the VAE decodes the final latent back into a visible image.

You usually never need to think about the VAE — each base model ships with one. But replacing it can fix specific issues:

| VAE | Problem it solves |
|---|---|
| `madebyollin/sdxl-vae-fp16-fix` | The official SDXL VAE produces NaN errors and black frames in fp16; this one is fp16-safe |
| `stabilityai/sd-vae-ft-mse` | The default SD 1.5 VAE causes colour bleeding and edge fringing; this reduces both |

**Rule:** swap the VAE only when you see visual artefacts that match its description. VAEs are family-specific — an SDXL VAE will not work on SD 1.5.

---

## LoRA (`model_type: lora`)

A **Low-Rank Adaptation** is a small patch applied on top of a base model's weights. It cannot run standalone — it modifies the base model during inference.

Two common uses:

- **Style LoRA** — shifts the output toward a visual style (anime, pixel art, photorealism). Controlled by a strength parameter (typically 0.5–1.0).
- **Speed LoRA** (e.g. LCM LoRA) — changes the scheduler behaviour to converge in 4–8 steps instead of 25–50. Does not alter visual style; requires pairing with `LCMScheduler` and low CFG (1–2).

`compatible_bases` lists which base families the LoRA was trained on. Loading a LoRA on the wrong base produces noise or garbage.

---

## ControlNet (`model_type: controlnet`)

Adds **structural guidance** to a base model. You provide a conditioning image (edge map, depth map, pose skeleton, etc.) and ControlNet forces the output to follow that structure while the base model handles style and details.

Each ControlNet is trained for a specific conditioning type AND a specific base family:

| Variant | What it reads | Use case |
|---|---|---|
| Canny | Edge map | Preserve outlines and hard structure |
| Depth | Depth map | Preserve spatial layout and foreground/background relationships |
| Lineart | Clean line drawing | Sketch-to-render workflows |
| OpenPose | Joint skeleton | Transfer human body pose from a reference image |
| Inpaint | Image + mask | Localised edits — only the masked area changes |
| Tile | Tiled image regions | Detail recovery and texture restoration |

ControlNet is heavier than T2I Adapter (adds a parallel UNet branch) but typically gives tighter structural adherence.

---

## T2I Adapter (`model_type: t2i_adapter`)

Functionally similar to ControlNet — provides structural conditioning — but architecturally lighter (feeds into the UNet encoder rather than running a full parallel branch). Faster and uses less VRAM; slightly less precise control.

Same family-compatibility rules as ControlNet apply.

---

## IP-Adapter (`model_type: ip_adapter`)

Lets you use an **image as a prompt** instead of (or alongside) text. The reference image drives composition, colour palette, and style. The base model still generates new pixels — it is not inpainting or copying.

Useful when you want to "paint in the style of" a reference or keep a character consistent across multiple generations. The IP-Adapter repo typically ships several variants (`sd15`, `sdxl`, `vit-h`, etc.) for different fidelity trade-offs.

---

## Upscalers (`model_type: upscaler`)

Take a small image and produce a larger one with recovered detail. Two technologies:

| Type | Examples | How it works |
|---|---|---|
| **Diffusion upscaler** | SD x4 Upscaler | Runs a diffusion process conditioned on the low-res input; accepts an optional text prompt to guide added detail. Slower, more creative. |
| **GAN upscaler** (ESRGAN) | UltraSharp 4x, RealESRGAN 4x+ | A generative adversarial network trained specifically on upscaling. Fast, deterministic, no prompt needed. |

ESRGAN upscalers are **family-agnostic** — they work on any image regardless of which model generated it. The diffusion upscaler is based on SD 2 and is more specialised.

---

## Face restoration (`model_type: face_restore`)

Post-processing tool that detects faces in an image, runs each face through a GAN trained on high-quality face data, then blends the result back into the original image. Applied after generation, not during.

Useful when the base model produces blurry or distorted faces, especially after upscaling.

---

## Vision-language models (`model_type: vision_language`)

Not a generative model — reads an image and outputs a text description. Used for automatic captioning, alt-text generation, and prompt reverse-engineering. Takes no text prompt as input (or an optional prefix).

---

## Quick decision guide

```
What do I need?
│
├─ Generate an image from a prompt          → base_diffusion (pick a family first)
│
├─ Generate + force a specific structure    → base_diffusion + controlnet or t2i_adapter
│
├─ Generate + steer style without retraining → base_diffusion + lora
│
├─ Generate + use an image as the style source → base_diffusion + ip_adapter
│
├─ Fix colour artefacts or black frames     → swap the vae
│
├─ Make the image bigger with more detail   → upscaler (ESRGAN for speed, SD x4 for creativity)
│
├─ Fix blurry or distorted faces            → face_restore (after generation)
│
└─ Get a text description of an image       → vision_language
```

---

## `compatible_bases` field

Every helper model (LoRA, ControlNet, T2I Adapter, VAE, IP-Adapter) has a `compatible_bases` list. This is the authoritative answer to "which base can I pair this with?" Values match the `base_model` field on base models (e.g. `sdxl`, `sd1.5`, `flux`).

Base models and standalone tools (upscalers, face restore, vision-language) have an empty `compatible_bases` because they do not depend on a base.
