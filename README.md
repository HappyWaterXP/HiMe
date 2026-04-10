# hime

Robot visual task execution server with planner/observer agents, optional memory, and a simple inference client.

## Requirements
- Python 3.13+
- `uv` (recommended) or `pip`

## Install
Using `uv` (recommended):
```bash
uv venv
uv sync
```


## Configure Environment
1. Create your env file
```bash
cp env.example.sh env.sh
```

2. Edit `env.sh`
- Set planner/observer/embedding endpoints, API keys, and model names.
- Choose prompt files via `PLANNER_PROMPT_NAME` and `OBSERVER_PROMPT_NAME`.
- Adjust optional runtime knobs if needed.

3. Load env
```bash
source ./env.sh
```

## Prompts
Prompt details and output formats are documented in `prompt/README.md`.

## Ablation Settings
Ablation profiles and tunable knobs are documented in `env.example.sh`.

## Client Integration (Required)
We provide reference clients in `widow/infer.py` and `flat_memory/flat_memory_client.py`. Use them as templates for your own robot policy code. You must adapt these files to match your real robot deployment (camera drivers, policy interface, and control loop). Keep the task server API calls, inputs, and outputs consistent, and replace the robot-specific parts with your implementation.

Task server API used by the client:
- `POST /tasks`: form fields `global_instruction`, `observer_window_size`, `human_intervene_for_planner`, `planner_execution_mode`; files `initial_image`, `initial_waist_image`; response `TaskPublicState` with fields `task_id`, `is_done`, `runtime_state`, `planner_status`, `plan_list`, `summary`, `current_subtask_description`
- `POST /tasks/{task_id}/step`: files `image[]` (main camera sequence), `waist_image[]` (waist camera sequence); response `TaskPublicState`
- `POST /tasks/{task_id}/user_instruction`: json `{ "user_new_instruction": "<text>" }`; response `TaskPublicState`

## Run the Server
```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port <PORT>
```

## Run the Inference Client (Widow)
```bash
uv run python widow/infer.py \
  --task_server_base_url <TASK_SERVER_BASE_URL> \
  --policy_host <POLICY_HOST> \
  --policy_port <POLICY_PORT> \
  --policy_trace_npz_folder ./_policy_traces
```

## Optional: Flat Memory Service
If you use the Flat Memory service, start it separately:
```bash
uv run uvicorn flat_memory.app:app --host 0.0.0.0 --port <PORT>
```
Configure it via the `FLAT_MEMORY_*` variables in `env.sh`.
