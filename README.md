# Stable Diffusion Lab

A **Vue 3 + TypeScript + Vuetify** frontend connected to a **Python FastAPI** backend for AI image generation. The backend loads Stable Diffusion models from [HuggingFace Hub](https://huggingface.co) and [CivitAI](https://civitai.com) and exposes a REST API for image creation through text prompts.

---

## Architecture

```
┌─────────────────────────────┐      REST API      ┌──────────────────────────────┐
│  Frontend  (Vue 3/Vuetify)  │ ◄────────────────► │  Backend  (FastAPI / Python) │
│  http://localhost:5173      │                    │  http://localhost:8000        │
└─────────────────────────────┘                    └──────────────────────────────┘
                                                              │
                                                    ┌─────────┴──────────┐
                                                    │   HuggingFace Hub  │
                                                    │   CivitAI          │
                                                    └────────────────────┘
```

## Features

- **Prompt form** – positive and negative prompts
- **Model selector** – pick from preset HuggingFace or CivitAI models, or enter a custom ID
- **Generation parameters** – width, height, steps, CFG scale, seed, number of images
- **On-demand model loading** – download & cache models before generation
- **Image gallery** – view, zoom, and download generated images
- **Backend status** – live indicator of the connected GPU/CPU device and loaded model

---

## Quick Start with Docker Compose

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/) v2
- *(Optional)* NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU acceleration

### Run

```bash
# Clone the repository
git clone https://github.com/Guebbit/stable-diffusion-lab.git
cd stable-diffusion-lab

# (Optional) provide your CivitAI API key for private/gated models
export CIVITAI_API_KEY=your_key_here

# Build and start all services
docker compose up --build
```

Open **http://localhost:5173** in your browser.

> **GPU support**: Uncomment the `deploy` section in `docker-compose.yml` to pass the NVIDIA GPU into the backend container.

---

## Development Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server at **http://localhost:5173** proxies `/api/*` requests to the backend at `http://localhost:8000`.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/status` | Backend health, device, and loaded model |
| `POST` | `/api/models/load` | Download and load a model |
| `POST` | `/api/generate` | Generate images from a prompt |

### `POST /api/generate` – request body

```json
{
  "prompt": "a beautiful landscape, oil painting",
  "negative_prompt": "blurry, low quality",
  "model_id": "runwayml/stable-diffusion-v1-5",
  "model_source": "huggingface",
  "width": 512,
  "height": 512,
  "num_inference_steps": 20,
  "guidance_scale": 7.5,
  "seed": null,
  "num_images": 1
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_CACHE_DIR` | `/app/models_cache` | Directory where model weights are cached |
| `CIVITAI_API_KEY` | *(empty)* | API key for downloading private CivitAI models |
