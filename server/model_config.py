from __future__ import annotations

"""
Single editable Python config for planner / observer / embedding model routing.

Usage:
- Edit the values below directly.
- Restart `uvicorn server.app:app` to apply changes.
"""

# Shared defaults (used when a section field is left empty)
OPENAI_API_KEY = "xx"
OPENAI_BASE_URL = "https://aigc.x-see.cn/v1"
VLM_MODEL = "qwen3-vl-30b-a3b-instruct"

# Planner (empty string means fallback to shared defaults above)
PLANNER_OPENAI_API_KEY = ""
PLANNER_OPENAI_BASE_URL = ""
PLANNER_VLM_MODEL = ""

# Observer (empty string means fallback to shared defaults above)
OBSERVER_OPENAI_API_KEY = ""
OBSERVER_OPENAI_BASE_URL = ""
OBSERVER_VLM_MODEL = ""

# Embedding (empty string means fallback to shared defaults above)
EMBEDDING_OPENAI_API_KEY = ""
EMBEDDING_OPENAI_BASE_URL = ""
EMBEDDING_MODEL = "text-embedding-3-large"
