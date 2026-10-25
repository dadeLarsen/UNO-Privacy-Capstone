"""Tasks, by name. Each task package owns its data, strategies, and evaluation.

A task may import ``privacy_bench.core`` and ``privacy_bench.models``, never another task.
"""

from .classification.task import ClassificationTask
from .rewrite.task import RewriteTask

TASKS = {task.name: task for task in (ClassificationTask(), RewriteTask())}


def get_task(name: str):
    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(f"Unknown task {name!r}; available: {', '.join(sorted(TASKS))}") from None
