# Copilot Instructions

## Coding Principles

- Keep code **SOLID** (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion).
- Keep code **DRY** (Don't Repeat Yourself).
- Keep code **KISS** (Keep It Simple, Stupid).

## Commenting Style

### TypeScript (Frontend)

- Comments should be **short and practical** — just enough to understand intent at a glance.
- ADHD-friendly: brief, scannable, no walls of text.
- Every function gets a one-liner explaining what it does.
- Variables with non-obvious purpose get a short inline comment.

### Python (Backend)

- Comments should be **detailed and thorough**, especially on AI/ML-related logic (pipelines, tensors, diffusion concepts, model loading).
- Explain the "why" behind decisions, not just the "what".
- Include short conceptual context for anyone unfamiliar with diffusion models or torch internals.
- Still ADHD-friendly: use whitespace, separators, and bullet points to keep things scannable.

## General

- Comments should explain the logic of every function and every variable that has some complexity attached.
- Prefer clarity over cleverness.
