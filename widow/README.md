# Widow Client

`widow/infer.py` is a reference client that connects your robot policy to the task server.

You must adapt it to your real robot deployment:
- camera drivers and image capture
- policy interface and control loop
- safety constraints and execution timing

Keep the task server API calls consistent with `server/app.py`.
