import re
from datetime import datetime
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

# Host-mounted directories
MODELS_DIR = Path("/models")
GENERATED_DIR = Path("/generated")
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_") or "model"


def _list_models() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    # Accept both folder-based HF models and single-file models.
    return sorted(item for item in MODELS_DIR.iterdir() if item.is_dir() or item.is_file())


def _pick_model(models: list[Path]) -> Path:
    print("\nAvailable models:")
    for idx, model_path in enumerate(models, start=1):
        print(f"  {idx}. {model_path.name}")

    while True:
        choice = input("\nChoose a model number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print("Invalid selection. Please enter one of the listed numbers.")


def _load_pipeline(model_path: Path):
    # Use GPU if available, otherwise run on CPU.
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    if model_path.is_dir():
        pipe = AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = AutoPipelineForText2Image.from_single_file(model_path, torch_dtype=dtype)

    return pipe.to("cuda" if use_cuda else "cpu")


def main() -> None:
    models = _list_models()
    if not models:
        print("No models found in /models. Run the downloader first.")
        return

    selected_model = _pick_model(models)
    prompt = input("Enter prompt: ").strip()
    if not prompt:
        print("Prompt cannot be empty.")
        return

    print(f"\nLoading model: {selected_model.name}")
    pipe = _load_pipeline(selected_model)

    print("Generating image...")
    image = pipe(prompt=prompt).images[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_label = _sanitize(selected_model.stem if selected_model.is_file() else selected_model.name)
    output_path = GENERATED_DIR / f"{model_label}_{timestamp}.png"
    image.save(output_path)

    print(f"Done. Image saved to: {output_path}")


if __name__ == "__main__":
    main()
