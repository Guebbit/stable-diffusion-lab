"""
Direct Python inference adapter — uses diffusers/transformers/torch directly.

This is the primary inference backend. It loads models into GPU/CPU memory
and runs inference in-process. Best for single-user local deployment.
"""
