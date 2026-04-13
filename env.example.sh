#!/usr/bin/env bash
# Example environment file.
# Usage:
#   cp env.example.sh env.sh
#   edit env.sh
#   source ./env.sh

# =====================
# Planner (required)
# =====================
# OpenAI-compatible endpoint and model for the planner.
# Required for task server. These must be valid for your provider.
# Example base URL: https://<host>/v1
export PLANNER_OPENAI_API_KEY=""
export PLANNER_OPENAI_BASE_URL=""
# Default: empty (must be set by user)
export PLANNER_VLM_MODEL=""

# =====================
# Observer (required)
# =====================
# OpenAI-compatible endpoint and model for the observer.
# The observer monitors recent frames and decides if replanning is needed.
export OBSERVER_OPENAI_API_KEY=""
export OBSERVER_OPENAI_BASE_URL=""
# Default: empty (must be set by user)
export OBSERVER_VLM_MODEL=""

# =====================
# Embedding (required when memory is enabled)
# =====================
# Used for memory retrieval. If you disable memory via ablation, these can be empty.
export EMBEDDING_OPENAI_API_KEY=""
export EMBEDDING_OPENAI_BASE_URL=""
# Default: empty (must be set when memory is enabled)
export EMBEDDING_MODEL=""
# 0 means auto-detect embedding dimension
export EMBEDDING_DIM="0"

# =====================
# Server behavior (optional)
# =====================
# ABLATION_PROFILE controls observer + memory behavior.
# Defaults to "hime" if empty/unknown.
# Options:
#   hime: full pipeline (observer + memory)
#   hime_wo_sentry: observer off, memory on
#   transient_memory: observer on, memory off
#   transient_memory_wo_sentry: observer off, memory off
#   only_image: memory uses images only (no text storage)
#   only_text: memory uses text only (no image storage)
#   no_management: disable memory update/delete (create/query only)
#   FIFO: fixed-size FIFO memory (requires max records)
# Default: hime
export ABLATION_PROFILE="hime"
# Prompt names under ./prompt (without .txt).
# Default values below are safe for initial testing.
# Default: task1 / task1_obs
export PLANNER_PROMPT_NAME="task1"
export OBSERVER_PROMPT_NAME="task1_obs"
# Memory policy override (optional).
# Values: allow_all | disable_all | query_create_only
# If empty, uses the ablation profile default.
# Default: empty (use ablation default)
export PLANNER_MEMORY_OP_POLICY=""
# Memory cap (integer). Empty means use profile defaults.
# Default: empty (use ablation default)
export PLANNER_MEMORY_MAX_RECORDS=""
# Planner fallback trigger if observer keeps returning not_done.
# Default 60 means: if 60 images pass without completion, force a planner step.
# Default: 60
export PLANNER_ROUND_FALLBACK_MAX_IMAGES="60"
# Observer sliding window size (number of recent frames).
# Larger values use more recent context but cost more tokens.
# Default: 5
export OBSERVER_WINDOW_SIZE="5"

# =====================
# Server resume (optional)
# =====================
# Point to an existing task folder under _server_data/task_*
# When set, the server resumes the task/memory state from disk.
# Default: empty (disabled)
export TASK_RESUME_DIR=""
export TASK_RESUME_ROUND=""
# Resume memory only (if not using TASK_RESUME_DIR).
# Used for warm-starting memory from logs.
# Default: empty (disabled)
export MEMORY_RESUME_DIR=""

# =====================
# Flat Memory (optional, separate service)
# =====================
# This is a separate HTTP service. Only needed if you run flat_memory/app.py.
# Default: empty (must be set when flat memory service is used)
export FLAT_MEMORY_OPENAI_API_KEY=""
export FLAT_MEMORY_OPENAI_BASE_URL=""
export FLAT_MEMORY_MODEL_NAME=""
# Prompt file name under ./prompt (without .txt).
# Empty means use the built-in default prompt.
# Default: empty (use built-in prompt)
export FLAT_MEMORY_PROMPT_NAME=""
# VLM timeout in seconds for flat memory selection.
# Default: 180
export FLAT_MEMORY_OPENAI_TIMEOUT_S="180"
# Merge distance d for 1D single-linkage clustering of nominations.
# Default: 5
export FLAT_MEMORY_MERGE_DISTANCE_D="5"
# Max selected keyframes kept in memory.
# Default: 8
export FLAT_MEMORY_MAX_KEYFRAMES="8"
# Max number of recent frames to consider per step.
# Default: 8
export FLAT_MEMORY_RECENT_MAX="8"
# Composite image layout for main+wrist: horizontal | vertical
# Default: horizontal
export FLAT_MEMORY_COMPOSITE_LAYOUT="horizontal"
# If >0, resize composite so the longer side is capped to this value.
# Default: 0 (no resize)
export FLAT_MEMORY_COMPOSITE_MAX_SIDE="0"
# Where to store flat memory logs (relative to repo root).
# Default: ./flat_memory_logs
export FLAT_MEMORY_LOG_ROOT="./flat_memory_logs"
# Save selected keyframes to disk (1=yes, 0=no).
# Default: 1 (save)
export FLAT_MEMORY_SAVE_KEYFRAMES="1"
# Save recent frames to disk (1=yes, 0=no).
# Default: 0 (do not save)
export FLAT_MEMORY_SAVE_RECENT_FRAMES="0"
# Save JSONL logs (1=yes, 0=no).
# Default: 1 (save)
export FLAT_MEMORY_LOG_JSONL="1"
# Optional: load history keyframes from previous tasks.
# Point to a folder containing task logs with events.jsonl.
# Default: empty (disabled)
export FLAT_MEMORY_HISTORY_MEMORY_DIR=""

echo "[env.example.sh] exported variables"
