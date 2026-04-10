# Prompts

This directory contains prompt templates for the planner and observer.

## Planner Outputs

### Plan-list mode (multi-step)
Used by:
- `task3.txt`
- `task3_hime_wo_sentry.txt`
- `task3_only_image.txt`
- `task3_only_text.txt`
- `task3_no_management.txt`
- `task3_FIFO.txt`

Output format (exact):
```xml
<summary>
...
</summary>

<memory_operations>
...
</memory_operations>

<plan_list>
...
</plan_list>
```

### Single-step mode (no memory)
Used by:
- `task3_transient_memory.txt`
- `task3_transient_memory_wo_sentry.txt`

Output format (exact):
- XML with `<summary>`, `<subtask>`, and `<is_complete>`.
- `<subtask>` is the single next action (Pick-And-Place format).
- No plan list, no memory operations.

## Observer Prompts
- `task1_obs.txt`, `task2_obs.txt`, `task3_obs.txt` are observer prompts.

## Task1 / Task2
- `task1.txt`, `task2.txt` are planner prompts for tasks 1/2.
