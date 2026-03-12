from __future__ import annotations

"""
Single editable Python config for planner / observer / embedding model routing.

Usage:
- Edit the values below directly.
- Restart `uvicorn server.app:app` to apply changes.
"""

# Planner (independent)
PLANNER_OPENAI_API_KEY = "xx-planner"
PLANNER_OPENAI_BASE_URL = "https://planner.example.com/v1"
PLANNER_VLM_MODEL = "qwen3-vl-30b-a3b-instruct"

# Observer (independent)
OBSERVER_OPENAI_API_KEY = "xx-observer"
OBSERVER_OPENAI_BASE_URL = "https://observer.example.com/v1"
OBSERVER_VLM_MODEL = "qwen3-vl-8b-instruct"

# Embedding (independent)
EMBEDDING_OPENAI_API_KEY = "xx-embedding"
EMBEDDING_OPENAI_BASE_URL = "https://embedding.example.com/v1"
EMBEDDING_MODEL = "text-embedding-3-large"
# Embedding vector dimension:
# - 0: auto infer from embedding API response dimension (recommended)
# - >0: enforce / normalize to fixed dimension
EMBEDDING_DIM = 0
