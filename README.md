# stable-diffusion-lab

Simple Docker setup with:
- a model downloader that saves into your host `./models` folder
- a basic terminal CLI (no GUI) that lets you choose a model and generate an image

## 1) Configure tokens (optional but recommended)
Create `.env` in the project root:

```env
HF_TOKEN=your_huggingface_token
CIV_TOKEN=your_civitai_token
```

## 2) Download models to host
This runs `download_models.py` in Docker and writes files to `./models` on your machine:

```bash
docker compose run --rm model-downloader
```

You can manually add/remove model files in `./models` at any time.

## 3) Generate an image from CLI
This starts an interactive terminal app. It will:
1. list available models from `./models`
2. ask you to pick one
3. ask for a prompt
4. save image in `./generated`

```bash
docker compose run --rm cli
```

Output filename format:

```text
<model_name>_<YYYYMMDD_HHMMSS>.png
```
