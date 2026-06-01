"""
Job orchestrator — queue management, worker dispatch, and state machine.

This package manages the lifecycle of background jobs:
- Polling the database for PENDING jobs
- Dispatching them to workers (asyncio tasks + thread pool)
- Managing the state machine transitions
- Broadcasting progress events to connected WebSocket clients
"""
