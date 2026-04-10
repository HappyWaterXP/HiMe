# Flat Memory Service

This service provides a keyframe-based memory store used by the planner.

## Run
```bash
uv run uvicorn flat_memory.app:app --host 0.0.0.0 --port <PORT>
```

## Configure
Use the `FLAT_MEMORY_*` variables in `env.sh` / `env.example.sh`.
