# Widow Client

`widow/infer.py` is a **reference client** that connects your robot policy to the task server.
It shows how to:
- send initial observations
- stream step observations
- parse the task server response
- turn the response into robot actions

This file is **not** plug-and-play. You must adapt it to your real robot deployment:
- camera drivers and image capture
- policy inference interface and control loop
- safety constraints, speed limits, and execution timing

Keep the task server API calls consistent with `server/app.py`.

## What This Client Does (High-Level Flow)
1. Capture the **initial** images (main + wrist/waist).
2. `POST /tasks` to create a task.
3. Loop:
   - capture new images
   - `POST /tasks/{task_id}/step`
   - read `current_subtask_description` / `plan_list` / `summary`
   - execute your robot policy

## Task Server API (Must Match)
The task server endpoints used here are:
- `POST /tasks`
  - form fields: `global_instruction`, `observer_window_size`, `human_intervene_for_planner`, `planner_execution_mode`
  - files: `initial_image`, `initial_waist_image`
  - response: `task_id`, `plan_list`, `summary`, `current_subtask_description`
- `POST /tasks/{task_id}/step`
  - files: repeated `image` (main camera), repeated `waist_image` (wrist camera)
  - response: updated `plan_list`, `summary`, `current_subtask_description`

If you change this file, keep these request/response formats consistent with `server/app.py`.

## Key Places To Adapt
- **Image capture**: replace the dummy/placeholder capture with your camera pipeline.
- **Robot policy**: replace the example policy call with your real policy inference.
- **Action execution**: map `current_subtask_description` into robot controls.
- **Timing**: control step rate to match your hardware and safety constraints.
- **Safety**: add guards for collision checks, workspace limits, and emergency stop.

## Common Gotchas
- If you send **only one** camera image, the server will error. Both main and wrist are required.
- If images are not RGB or not PNG/JPEG, the server may reject them.
- If you stream too fast, you may overload your policy or the VLM; throttle if needed.

## Minimal Usage Example
```bash
source ./env.sh
uv run python widow/infer.py --task_server_base_url http://127.0.0.1:8000
```

That command will not run your real robot until you replace the stubs in `widow/infer.py`.
