/**
 * Curated catalog of Stable Diffusion models available for selection.
 *
 * Each entry doubles as a dropdown option (id + name + description) AND
 * a catalog card (longDescription + tags + sourceUrl + family).
 *
 * family:
 *   'sd15' → Stable Diffusion 1.x / 2.x family  (512–768 px native, lighter)
 *   'sdxl' → Stable Diffusion XL family          (1024 px native, heavier)
 *
 * The backend auto-detects the family from the model ID string, so this
 * field is informational for the UI only (filters, badges, warnings).
 */

import type { ModelOption } from '../types'

// ─── HuggingFace models ────────────────────────────────────────────────────
//
// IDs are HuggingFace repo slugs in the form "org/repo".
// from_pretrained() downloads the full model repo on first use.

export const huggingfaceModels: ModelOption[] = [
  {
    id: 'runwayml/stable-diffusion-v1-5',
    name: 'Stable Diffusion v1.5',
    source: 'huggingface',
    family: 'sd15',
    description: 'The classic SD 1.5 base — fast, widely compatible',
    tags: ['general', 'fast', 'classic'],
    longDescription:
      'The canonical Stable Diffusion 1.5 checkpoint, fine-tuned by Runway from the CompVis SD 1.4 base. ' +
      'Native resolution is 512 × 512. It is the most widely supported model: virtually every ' +
      'LoRA, ControlNet, and community extension targets it. A good all-purpose starting point.',
    sourceUrl: 'https://huggingface.co/runwayml/stable-diffusion-v1-5',
  },
  {
    id: 'stabilityai/stable-diffusion-2-1',
    name: 'Stable Diffusion v2.1',
    source: 'huggingface',
    family: 'sd15',
    description: 'SD 2.1 — improved anatomy, 768 px native resolution',
    tags: ['general', 'detailed'],
    longDescription:
      'Stability AI\'s second-generation base model. Trained with a new OpenCLIP text encoder ' +
      '(instead of CLIP), which handles long, complex prompts better. Native resolution is ' +
      '768 × 768. Note: LoRAs trained for SD 1.5 are NOT compatible with this model family.',
    sourceUrl: 'https://huggingface.co/stabilityai/stable-diffusion-2-1',
  },
  {
    id: 'CompVis/stable-diffusion-v1-4',
    name: 'Stable Diffusion v1.4',
    source: 'huggingface',
    family: 'sd15',
    description: 'The original SD 1.4 by CompVis — lightweight and historical',
    tags: ['classic', 'fast', 'lightweight'],
    longDescription:
      'The original public release of Stable Diffusion by CompVis (LMU Munich). ' +
      'Trained on LAION-Aesthetics v2 at 512 × 512. Lighter than 1.5 and useful for ' +
      'historical comparisons or resource-limited machines. Most community content now ' +
      'targets v1.5 instead.',
    sourceUrl: 'https://huggingface.co/CompVis/stable-diffusion-v1-4',
  },
  {
    id: 'prompthero/openjourney',
    name: 'OpenJourney v4',
    source: 'huggingface',
    family: 'sd15',
    description: 'Midjourney-inspired style — vivid, painterly outputs',
    tags: ['artistic', 'stylized', 'midjourney'],
    longDescription:
      'A fine-tune of SD 1.5 on Midjourney v4 outputs by PromptHero. ' +
      'Trigger the style with the prefix "mdjrny-v4 style" in your prompt. ' +
      'Produces vivid, painterly images with the characteristic Midjourney look ' +
      '— great for concept art and stylized illustrations.',
    sourceUrl: 'https://huggingface.co/prompthero/openjourney',
  },
  {
    id: 'dreamlike-art/dreamlike-photoreal-2.0',
    name: 'Dreamlike Photoreal 2.0',
    source: 'huggingface',
    family: 'sd15',
    description: 'Photorealistic fine-tune — detailed skin tones and lighting',
    tags: ['photorealistic', 'portraits', 'detailed'],
    longDescription:
      'A photorealism-focused fine-tune of SD 1.5 by Dreamlike Art. ' +
      'Excels at realistic portraits, landscapes, and product shots. ' +
      'Best results at 768 × 512 or higher. Add "photo" or "photograph" in the ' +
      'prompt to steer the model toward photographic output.',
    sourceUrl: 'https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0',
  },
  {
    id: 'Lykon/dreamshaper-8',
    name: 'DreamShaper 8',
    source: 'huggingface',
    family: 'sd15',
    description: 'Highly versatile — photos, art, fantasy at 512 px',
    tags: ['versatile', 'photorealistic', 'artistic'],
    longDescription:
      'DreamShaper 8 by Lykon is one of the most popular community fine-tunes of SD 1.5. ' +
      'Covers a wide range of styles: photorealistic portraits, fantasy art, and concept design. ' +
      'Also available as a CivitAI checkpoint with the same weights. ' +
      'Works best with DPM++ 2M Karras sampler at 20–30 steps.',
    sourceUrl: 'https://huggingface.co/Lykon/dreamshaper-8',
  },
  {
    id: 'stabilityai/stable-diffusion-xl-base-1.0',
    name: 'Stable Diffusion XL 1.0',
    source: 'huggingface',
    family: 'sdxl',
    description: 'SDXL base — high detail and color fidelity at 1024 px',
    tags: ['high-quality', 'detailed', 'large'],
    longDescription:
      'Stability AI\'s SDXL 1.0 base model. Native resolution is 1024 × 1024. ' +
      'Uses a two-stage architecture (base + optional refiner) and a much larger ' +
      'U-Net than SD 1.5, producing significantly more detailed and coherent images. ' +
      'Requires more VRAM (≈8 GB for float16). Recommended GPU: RTX 3080 or better.',
    sourceUrl: 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0',
  },
  {
    id: 'stabilityai/sdxl-turbo',
    name: 'SDXL Turbo',
    source: 'huggingface',
    family: 'sdxl',
    description: 'Distilled SDXL — near-instant results in 1–4 steps',
    tags: ['fast', 'real-time', 'sdxl'],
    longDescription:
      'SDXL Turbo uses Adversarial Diffusion Distillation (ADD) to compress ' +
      'the full SDXL sampling process into 1–4 steps without a major quality loss. ' +
      'Ideal for rapid iteration and real-time applications. ' +
      'Use guidance_scale=0 (or very low) and num_inference_steps=1–4 for best results. ' +
      'Not recommended for sketch2ink (ControlNet needs more steps).',
    sourceUrl: 'https://huggingface.co/stabilityai/sdxl-turbo',
  },
]

// ─── CivitAI models ────────────────────────────────────────────────────────
//
// IDs are CivitAI MODEL VERSION IDs (the numeric ID in the "modelVersionId"
// query param on civitai.com).  You can find the version ID in the URL:
//   https://civitai.com/models/4384?modelVersionId=128713
//                                                   ^^^^^^
// The backend validates that these are numeric-only strings before downloading.

export const civitaiModels: ModelOption[] = [
  {
    id: '128713',
    name: 'DreamShaper 8',
    source: 'civitai',
    family: 'sd15',
    description: 'Versatile art/photo model — version 128713',
    tags: ['versatile', 'photorealistic', 'artistic'],
    longDescription:
      'DreamShaper 8 by Lykon (CivitAI version 128713). ' +
      'One of the most downloaded models on CivitAI. Handles portraits, fantasy, ' +
      'concept art and landscapes equally well. ' +
      'For best results use DPM++ 2M Karras, 20–30 steps, CFG 4–7.',
    sourceUrl: 'https://civitai.com/models/4384?modelVersionId=128713',
  },
  {
    id: '130072',
    name: 'Realistic Vision V5.1',
    source: 'civitai',
    family: 'sd15',
    description: 'Hyper-photorealistic portraits and scenes — version 130072',
    tags: ['photorealistic', 'portraits', 'detailed'],
    longDescription:
      'Realistic Vision V5.1 by SG_161222 (CivitAI version 130072). ' +
      'Focused on extreme photorealism — skin pores, fabric textures, natural lighting. ' +
      'Pairs well with VAE: vae-ft-mse-840000-ema-pruned.safetensors. ' +
      'Negative prompt should include "cartoon, painting, illustration" to steer away from art styles.',
    sourceUrl: 'https://civitai.com/models/4201?modelVersionId=130072',
  },
  {
    id: '403131',
    name: 'majicMIX Realistic v7',
    source: 'civitai',
    family: 'sd15',
    description: 'Asian-beauty–focused photorealism — version 403131',
    tags: ['photorealistic', 'portraits', 'asian-style'],
    longDescription:
      'majicMIX Realistic v7 by Merjic (CivitAI version 403131). ' +
      'Specialised in realistic East Asian portrait photography. ' +
      'Renders fine facial details, hair strands and natural skin tones exceptionally well. ' +
      'Works well at 512 × 768 (portrait) with 25–35 steps.',
    sourceUrl: 'https://civitai.com/models/43331?modelVersionId=403131',
  },
]
