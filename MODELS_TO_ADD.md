Right now, after the container is created, the backend is empty. It has no models and they have to be manually inserted.
I'd like a way to have these models already inserted in the database (NOT PREDOWNLOADED), maybe through a file that export a constant with the array of objects that describe the models.
These objects should have "download" boolean flag. If true: download after the container creation



Here a list of the models I'd like to add  (they are from another prompt, need to be converted in objects)

🎨 IMAGE GENERATION MODELS (4090 SAFE)
1. FLUX.1-dev
   ID: black-forest-labs/FLUX.1-dev
   Source: HuggingFace
   Tags: photorealism, prompt-adherence, high-quality, general
   Desc: High-end image generator with excellent realism and composition. Best open model for near-commercial image quality. Needs 20–24GB VRAM but runs fine on 4090 with optimizations.
   URL: https://huggingface.co/black-forest-labs/FLUX.1-dev
2. FLUX.1-schnell (FAST)
   ID: black-forest-labs/FLUX.1-schnell
   Source: HuggingFace
   Tags: ultra-fast, preview, low-steps, realtime
   Desc: Same family as FLUX-dev but optimized for speed. Great for iterative UI or prompt testing. Slight quality loss but extremely fast.
   URL: https://huggingface.co/black-forest-labs/FLUX.1-schnell
3. SDXL 1.0 Base
   ID: stabilityai/stable-diffusion-xl-base-1.0
   Source: HuggingFace
   Tags: general, versatile, ecosystem, balanced
   Desc: Most widely supported modern image model. Huge LoRA ecosystem (Civitai). Best “default stable pipeline” model.
   URL: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
4. SDXL Turbo (FASTEST SDXL)
   ID: stabilityai/sdxl-turbo
   Source: HuggingFace
   Tags: realtime, ultra-fast, preview, low-steps
   Desc: Extreme speed SDXL variant (1–4 steps). Great for testing pipelines or interactive apps.
   URL: https://huggingface.co/stabilityai/sdxl-turbo
5. Juggernaut XL (Civitai favorite)
   ID: civitai:288982 (varies by version)
   Source: Civitai
   Tags: photorealism, cinematic, SDXL finetune, portrait
   Desc: One of the best SDXL finetunes for realistic humans and cinematic lighting. Very popular production checkpoint.
   URL: https://civitai.com/models/133005/juggernaut-xl
6. DreamShaper XL
   ID: civitai:126688
   Source: Civitai
   Tags: stylized, semi-real, illustration, general
   Desc: Balanced SDXL model for artistic + semi-real outputs. Great general-purpose creative model.
   URL: https://civitai.com/models/112902/dreamshaper-xl
   ✏️ IMAGE EDITING / INPAINTING
7. FLUX Kontext (editing)
   ID: black-forest-labs/flux-kontext-dev
   Source: HuggingFace
   Tags: instruction-editing, consistency, object-edit, photoreal
   Desc: Strong prompt-based image editing while preserving identity and composition. Best modern editing model.
   URL: https://huggingface.co/black-forest-labs/flux-kontext-dev
8. SDXL Inpainting
   ID: runwayml/stable-diffusion-inpainting
   Source: HuggingFace
   Tags: inpainting, mask-edit, legacy, flexible
   Desc: Classic inpainting model. Less intelligent than FLUX but extremely compatible with tools and ControlNet workflows.
   URL: https://huggingface.co/runwayml/stable-diffusion-inpainting
   🧾 IMAGE UNDERSTANDING (VISION)
9. Qwen2.5-VL
   ID: Qwen/Qwen2.5-VL-7B-Instruct
   Source: HuggingFace
   Tags: OCR, reasoning, multimodal, documents
   Desc: Excellent vision-language model. Reads charts, UI, screenshots, text-heavy images reliably.
   URL: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
10. InternVL2
    ID: OpenGVLab/InternVL2-8B
    Source: HuggingFace
    Tags: vision QA, grounding, multimodal, reasoning
    Desc: Strong alternative to Qwen VL, especially good at spatial reasoning and complex image understanding.
    URL: https://huggingface.co/OpenGVLab/InternVL2-8B
    🎬 VIDEO GENERATION (4090 SAFE)
11. Wan 2.1 (small/quantized)
    ID: wan-ai/Wan2.1-T2V-1.3B (or similar small variants)
    Source: HuggingFace
    Tags: text-to-video, motion, realistic, balanced
    Desc: Best open video model family. Large variants are heavy, but 1–3B versions run locally with optimizations.
    URL: https://huggingface.co/wan-ai/Wan2.1-T2V-1.3B
12. CogVideoX-2B
    ID: THUDM/CogVideoX-2b
    Source: HuggingFace
    Tags: video, text-to-video, stable, general
    Desc: Solid lightweight video generation model. Easier to run than Wan large models, good for dev/testing.
    URL: https://huggingface.co/THUDM/CogVideoX-2b
13. LTX-Video (FASTEST)
    ID: Lightricks/LTX-Video
    Source: HuggingFace
    Tags: ultra-fast, preview, real-time-ish, low-quality
    Desc: Extremely fast video generation for testing pipelines. Not cinematic but ideal for iteration speed.
    URL: https://huggingface.co/Lightricks/LTX-Video
    ✏️ BONUS: INK / SKETCH MODELS
14. SDXL ControlNet Lineart
    ID: diffusers/controlnet-sdxl-lineart
    Source: HuggingFace
    Tags: sketch-to-image, lineart, controlnet, illustration
    Desc: Converts sketches/ink drawings into structured images. Best general sketch pipeline.
    URL: https://huggingface.co/diffusers/controlnet-sdxl-lineart
15. SD 1.5 ControlNet Scribble (FASTEST sketch)
    ID: lllyasviel/control_v11p_sd15_scribble
    Source: HuggingFace
    Tags: scribble, fast, control, anime-friendly
    Desc: Very fast sketch-to-image model. Lower quality than SDXL but excellent responsiveness and huge LoRA ecosystem.
    URL: https://huggingface.co/lllyasviel/control_v11p_sd15_scribble