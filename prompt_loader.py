# src/prompt_loader.py
import os
from functools import lru_cache

# 当前文件所在目录：.../memory/src
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# prompt 目录：.../memory/src/prompt
PROMPT_DIR = os.path.join(_THIS_DIR, "prompt")


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """
    从 src/prompt 目录加载 prompt 文本文件。
    例如:
        load_prompt("observer") -> 读取 observer.txt
        load_prompt("planner")  -> 读取 planner.txt
    """
    filename = f"{name}.txt"
    path = os.path.join(PROMPT_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()