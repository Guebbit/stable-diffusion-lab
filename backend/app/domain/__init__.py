"""
Domain layer — shared types, enums, value objects, and protocols.

This package is the "vocabulary" of the system. Every layer can import from here,
but this package never imports from any other layer. It defines:

- Enums and literal types (model sources, job states, generation tasks)
- Value objects (immutable data containers)
- Domain protocols (interfaces that adapters must implement)
- Base entity types used across layers
"""
