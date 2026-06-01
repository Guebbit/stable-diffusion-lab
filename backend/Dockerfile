# AI Lab Backend
#
# Multi-stage build: dependencies first (cached), then app code.
# This means code changes don't re-download all pip packages.

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for torch, Pillow, and PostgreSQL client
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

COPY . .

# Create storage directories
RUN mkdir -p /app/storage/models/huggingface \
             /app/storage/models/civitai \
             /app/storage/models/local \
             /app/storage/artifacts \
             /app/storage/thumbnails \
             /app/storage/temp \
             /app/storage/imports \
             /app/storage/logs

ENV PYTHONUNBUFFERED=1
ENV STORAGE_ROOT=/app/storage

EXPOSE 8000

# Entry point uses the new app package structure
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
