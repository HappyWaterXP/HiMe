from __future__ import annotations

"""
Single editable Python config for planner / observer / embedding model routing.

Usage:
- Edit the values below directly.
- Restart `uvicorn server.app:app` to apply changes.
"""

# Planner (independent)
PLANNER_OPENAI_API_KEY = ""
PLANNER_OPENAI_BASE_URL = ""
PLANNER_VLM_MODEL = ""

# Observer (independent)
OBSERVER_OPENAI_API_KEY = ""
OBSERVER_OPENAI_BASE_URL = ""
OBSERVER_VLM_MODEL = ""

# Embedding (independent)
EMBEDDING_OPENAI_API_KEY = ""
EMBEDDING_OPENAI_BASE_URL = ""
EMBEDDING_MODEL = ""
# Embedding vector dimension:
# - 0: auto infer from embedding API response dimension (recommended)
# - >0: enforce / normalize to fixed dimension
EMBEDDING_DIM = 0
