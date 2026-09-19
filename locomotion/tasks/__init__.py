"""The project-owned walking task."""

from locomotion.tasks.walking import (
    LOCOMOTION_TASK_IDS,
    WALKING_TASK_ID,
    LocomotionTask,
    WalkingCommand,
    WalkingCommandCfg,
    get_locomotion_task,
    locomotion_tasks,
    register_leggybot_tasks,
    walking_env_cfg,
)

__all__ = (
    "LOCOMOTION_TASK_IDS",
    "WALKING_TASK_ID",
    "LocomotionTask",
    "WalkingCommand",
    "WalkingCommandCfg",
    "get_locomotion_task",
    "locomotion_tasks",
    "register_leggybot_tasks",
    "walking_env_cfg",
)
