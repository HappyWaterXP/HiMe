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
# Fill with your provider settings.
export PLANNER_OPENAI_API_KEY=""
export PLANNER_OPENAI_BASE_URL=""
export PLANNER_VLM_MODEL=""

# =====================
# Observer (required)
# =====================
# OpenAI-compatible endpoint and model for the observer.
export OBSERVER_OPENAI_API_KEY=""
export OBSERVER_OPENAI_BASE_URL=""
export OBSERVER_VLM_MODEL=""

# =====================
# Embedding (required when memory is enabled)
# =====================
export EMBEDDING_OPENAI_API_KEY=""
export EMBEDDING_OPENAI_BASE_URL=""
export EMBEDDING_MODEL=""
export EMBEDDING_DIM="0"  # 0 means auto

# =====================
# Server behavior (optional)
# =====================
# ABLATION_PROFILE (controls observer + memory behavior)
#   hime: full pipeline (observer + memory)
#   hime_wo_sentry: observer off, memory on
#   transient_memory: observer on, memory off
#   transient_memory_wo_sentry: observer off, memory off
#   only_image: memory uses images only
#   only_text: memory uses text only
#   no_management: memory without delete/update management
#   FIFO: FIFO memory (fixed-size)
export ABLATION_PROFILE="hime"
# Prompt names under ./prompt (without .txt)
# Examples: task1, task2, task3
export PLANNER_PROMPT_NAME="task1"
export OBSERVER_PROMPT_NAME="task1_obs"
# Memory policy override (e.g. allow_all, disable_all, query_create_only)
export PLANNER_MEMORY_OP_POLICY=""
# Memory cap (integer). Empty means use profile defaults.
export PLANNER_MEMORY_MAX_RECORDS=""
# Planner fallback trigger if observer keeps returning not_done
export PLANNER_ROUND_FALLBACK_MAX_IMAGES="60"
# Observer sliding window size (number of recent frames)
export OBSERVER_WINDOW_SIZE="5"

# =====================
# Server resume (optional)
# =====================
# Point to an existing task folder under _server_data/task_*
export TASK_RESUME_DIR=""
export TASK_RESUME_ROUND=""
# Resume memory only (if not using TASK_RESUME_DIR)
export MEMORY_RESUME_DIR=""

# =====================
# Flat Memory (optional, separate service)
# =====================
export FLAT_MEMORY_OPENAI_API_KEY=""
export FLAT_MEMORY_OPENAI_BASE_URL=""
export FLAT_MEMORY_MODEL_NAME=""
# Prompt file name under ./prompt (without .txt)
export FLAT_MEMORY_PROMPT_NAME=""
export FLAT_MEMORY_OPENAI_TIMEOUT_S="180"
export FLAT_MEMORY_MERGE_DISTANCE_D="5"
export FLAT_MEMORY_MAX_KEYFRAMES="8"
export FLAT_MEMORY_RECENT_MAX="8"
export FLAT_MEMORY_COMPOSITE_LAYOUT="horizontal"
export FLAT_MEMORY_COMPOSITE_MAX_SIDE="0"
export FLAT_MEMORY_LOG_ROOT="./flat_memory_logs"
export FLAT_MEMORY_SAVE_KEYFRAMES="1"
export FLAT_MEMORY_SAVE_RECENT_FRAMES="0"
export FLAT_MEMORY_LOG_JSONL="1"
export FLAT_MEMORY_HISTORY_MEMORY_DIR=""

echo "[env.example.sh] exported variables"
