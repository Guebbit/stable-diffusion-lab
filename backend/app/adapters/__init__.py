"""
Adapter layer — inference execution implementations.

Each sub-package contains a concrete implementation of the domain protocols.
Adapters are the ONLY place that imports heavy ML libraries (torch, diffusers, etc.).

Available backends:
- direct/  → Direct Python inference using diffusers/transformers/torch
- bentoml/ → Delegates to a running BentoML service
- comfyui/ → Submits workflows to a ComfyUI server
"""
