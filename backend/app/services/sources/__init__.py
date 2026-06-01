"""
Model source providers — adapters for downloading models from external sources.

Each provider implements the ModelSourceProvider protocol, handling:
- Model metadata resolution (file lists, sizes, checksums)
- File downloads with resume support
- Source-specific authentication and rate limiting
"""

from app.services.sources.huggingface import HuggingFaceSourceProvider
from app.services.sources.civitai import CivitaiSourceProvider
from app.services.sources.local import LocalSourceProvider

__all__ = ["CivitaiSourceProvider", "HuggingFaceSourceProvider", "LocalSourceProvider"]
