# Stable Diffusion Lab – Backend

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by torch / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Directory for model checkpoints
RUN mkdir -p /app/models_cache

ENV MODELS_CACHE_DIR=/app/models_cache
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
