"""
API layer — FastAPI routers, request/response schemas, and WebSocket hub.

This layer handles HTTP concerns only:
- Route definitions and grouping
- Request validation via Pydantic schemas
- Response serialization
- WebSocket connection management

NO business logic, database queries, or ML library imports belong here.
"""
