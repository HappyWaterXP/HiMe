"""FastAPI HTTP server with ablation-awareness patch.

Differences from the original app.py:
1. Exposes ABLATION_STUDY via GET /ablation_config (read-only)
2. Any future ablation-specific routes or middleware can be added here without touching the core app.py logic.
"""
from fastapi import FastAPI
from .ablation_logic import get_ablation_config, AblationConfig

# Re-export the original app object (if you prefer not to duplicate routes)
# Alternatively, this module can be used as a decorator around src.server.app:app
# For simplicity, we just provide a minimal patch module.

def patch_app_with_ablation(app: FastAPI) -> FastAPI:
    """
    Add ablation-related endpoints to the existing app.

    New endpoints:
        GET /ablation_config
            Returns:
                {
                    "planner_mode": "plan_list" | "single_subtask",
                    "study": "<ABLATION_STUDY env value>"
                }
    """
    @app.get("/ablation_config")
    async def get_ablation_config_endpoint():
        cfg = get_ablation_config()
        study = os.environ.get("ABLATION_STUDY", "baseline")
        return {
            "planner_mode": cfg.planner_mode,
            "study": study,
        }

    return app