"""Shared mjlab task loading and sim-to-real configuration for both trainers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from locomotion import DEFAULT_TASK_ID

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnvCfg


def register_tasks() -> None:
    """Register every implemented locomotion task."""
    from locomotion.tasks import register_leggybot_skill_tasks

    register_leggybot_skill_tasks()


# Compatibility for existing callers; new code should use the more accurate name.
register_builtin_tasks = register_tasks


def make_env_cfg(
    task_id: str = DEFAULT_TASK_ID,
    *,
    num_envs: int,
    seed: int,
    play: bool = False,
    explicit_resets: bool = False,
    sim2real_randomization: bool = True,
) -> "ManagerBasedRlEnvCfg":
    """Load one task configuration with a deployment-compatible actor ABI.

    The actor intentionally does not receive simulator-perfect base linear
    velocity.  The critic retains it as privileged information.  If the physical
    robot has a deployed velocity estimator, add its estimate as an actor input in
    both simulation and deployment rather than restoring ground truth here.
    """
    register_tasks()

    from mjlab.envs import mdp as env_mdp
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task_id, play=play)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.auto_reset = not explicit_resets

    actor_cfg = cfg.observations["actor"]
    actor_cfg.terms.pop("base_lin_vel", None)

    # A task can own its robot model and sim-to-real randomization. Applying
    # generic velocity-task events there would duplicate or conflict with the
    # task definition, regardless of robot or skill name.
    if getattr(cfg, "manages_sim2real_randomization", False):
        cfg.sim2real_randomization_enabled = bool(
            sim2real_randomization and not play
        )
        if not cfg.sim2real_randomization_enabled:
            cfg.events.pop("domain_randomization", None)
            robot_cfg = cfg.scene.entities["robot"]
            if robot_cfg.articulation is not None:
                for actuator_cfg in robot_cfg.articulation.actuators:
                    actuator_cfg.delay_min_lag = 0
                    actuator_cfg.delay_max_lag = 0
        return cfg

    if play or not sim2real_randomization:
        return cfg

    # The upstream velocity task already randomizes friction, encoder bias, base
    # COM and external pushes.  These additions cover inertia and motor mismatch.
    cfg.events["link_inertia"] = EventTermCfg(
        mode="startup",
        func=env_mdp.dr.pseudo_inertia,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
            "alpha_range": (-0.05, 0.05),
            "d_range": (-0.025, 0.025),
        },
    )
    cfg.events["motor_pd_gains"] = EventTermCfg(
        mode="startup",
        func=env_mdp.dr.pd_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot", actuator_names=[".*"]),
            "kp_range": (0.9, 1.1),
            "kd_range": (0.9, 1.1),
            "operation": "scale",
        },
    )
    cfg.events["motor_effort_limits"] = EventTermCfg(
        mode="startup",
        func=env_mdp.dr.effort_limits,
        params={
            "asset_cfg": SceneEntityCfg("robot", actuator_names=[".*"]),
            "effort_limit_range": (0.9, 1.05),
            "operation": "scale",
        },
    )

    robot_cfg = cfg.scene.entities["robot"]
    if robot_cfg.articulation is not None:
        for actuator_cfg in robot_cfg.articulation.actuators:
            # 0--10 ms at the reference 200 Hz physics rate.
            actuator_cfg.delay_min_lag = 0
            actuator_cfg.delay_max_lag = 2
            actuator_cfg.delay_hold_prob = 0.8
            actuator_cfg.delay_update_period = 20

    return cfg
