from types import SimpleNamespace

import pytest
import torch

from locomotion.task import make_env_cfg, register_tasks
from locomotion.task_runner import TaskStampedOnPolicyRunner
from locomotion.tasks.skills import (
    ANIMATED_REFERENCE_WALK_TASK_ID,
    CONSERVATIVE_WALK_TASK_ID,
    PERIODIC_WALK_TASK_ID,
    QUASI_OMNIDIRECTIONAL_WALKING_TASK_ID,
    QUASI_OMNIDIRECTIONAL_WALKING_V2_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_V3_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_V4_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_V5_TASK_ID,
    PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID,
    QUASI_OMNIDIRECTIONAL_V2_REWARD_WEIGHTS,
    QUASI_OMNIDIRECTIONAL_V2_ACTUATOR_EFFORT_JOINT_WEIGHTS,
    STRAIGHT_PERIODIC_WALK_TASK_ID,
    SYMMETRIC_CYCLE_PERIODIC_WALK_TASK_ID,
    SYMMETRIC_PERIODIC_WALK_TASK_ID,
    UPDATED_DEFAULT_LATERAL_WALKING_TASK_ID,
    UPDATED_DEFAULT_WALKING_TASK_ID,
    DiscreteSagittalCommandCfg,
    QuasiOmnidirectionalCommand,
    QuasiOmnidirectionalCommandCfg,
    QuasiOmnidirectionalMixtureCommand,
    QuasiOmnidirectionalMixtureCommandCfg,
    SAGITTAL_TASK_ID,
    STAND_TASK_ID,
    STEP_IN_PLACE_TASK_ID,
    UNLOADING_WALK_TASK_ID,
    LOCOMOTION_TASK_IDS,
    WEIGHT_TRANSFER_TASK_ID,
)
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from locomotion.robot_profiles import LEGGYBOT_FIXED_ANKLE_PROFILE
from locomotion.tasks.leggybot import get_leggybot_robot_cfg
from locomotion.tasks.gait import (
    lateral_support_balance,
    lateral_velocity_command_range_curriculum,
    quasi_omnidirectional_command_curriculum,
    StagedQuasiOmnidirectionalCurriculum,
    velocity_command_range_curriculum,
)


def test_skill_tasks_register_with_shared_policy_abi() -> None:
    register_tasks()
    assert all(task in list_tasks() for task in LOCOMOTION_TASK_IDS)
    cfgs = [make_env_cfg(task, num_envs=2, seed=1) for task in LOCOMOTION_TASK_IDS]
    assert len({tuple(c.observations["actor"].terms) for c in cfgs}) == 1
    assert len({tuple(c.observations["critic"].terms) for c in cfgs}) == 1
    assert all("gait_clock" in c.observations["actor"].terms for c in cfgs)
    assert all("domain_randomization" in c.events for c in cfgs)
    assert all(c.sim2real_randomization_enabled for c in cfgs)


def test_new_leggybot_stand_is_the_default_task() -> None:
    cfg = make_env_cfg(num_envs=1, seed=1)
    assert cfg.commands["twist"].ranges.lin_vel_x == (0.0, 0.0)


def test_leggybot_xml_actuators_are_individually_addressable() -> None:
    cfg = get_leggybot_robot_cfg(ankle_pitch_deg=8.0)
    assert cfg.articulation is not None
    assert [actuator.target_names_expr for actuator in cfg.articulation.actuators] == [
        (joint_name,) for joint_name in LEGGYBOT_FIXED_ANKLE_PROFILE.joint_names
    ]


def test_skill_command_progression_is_explicit() -> None:
    assert make_env_cfg(STAND_TASK_ID, num_envs=1, seed=1).commands["twist"].ranges.lin_vel_x == (0.0, 0.0)
    assert make_env_cfg(WEIGHT_TRANSFER_TASK_ID, num_envs=1, seed=1).commands["twist"].ranges.lin_vel_x == (0.0, 0.0)
    assert make_env_cfg(STEP_IN_PLACE_TASK_ID, num_envs=1, seed=1).commands["twist"].ranges.lin_vel_x == (0.0, 0.0)
    assert make_env_cfg(ANIMATED_REFERENCE_WALK_TASK_ID, num_envs=1, seed=1).commands["twist"].ranges.lin_vel_x == (0.08, 0.08)
    assert make_env_cfg(CONSERVATIVE_WALK_TASK_ID, num_envs=1, seed=1).commands["twist"].ranges.lin_vel_x == (0.08, 0.08)
    assert make_env_cfg(UNLOADING_WALK_TASK_ID, num_envs=1, seed=1).commands["twist"].ranges.lin_vel_x == (0.08, 0.08)
    assert make_env_cfg(PERIODIC_WALK_TASK_ID, num_envs=1, seed=1).commands["twist"].ranges.lin_vel_x == (0.08, 0.08)
    assert make_env_cfg(STRAIGHT_PERIODIC_WALK_TASK_ID, num_envs=1, seed=1).commands["twist"].ranges.lin_vel_x == (0.08, 0.08)
    assert make_env_cfg(SYMMETRIC_CYCLE_PERIODIC_WALK_TASK_ID, num_envs=1, seed=1).commands["twist"].ranges.lin_vel_x == (0.08, 0.08)
    assert make_env_cfg(SYMMETRIC_PERIODIC_WALK_TASK_ID, num_envs=1, seed=1).commands["twist"].ranges.lin_vel_x == (0.08, 0.08)
    assert make_env_cfg(SAGITTAL_TASK_ID, num_envs=1, seed=1).commands["twist"].ranges.lin_vel_x == (-0.5, 0.5)


def test_short_walk_tasks_are_contact_conservative_and_pitch_stable() -> None:
    animated = make_env_cfg(ANIMATED_REFERENCE_WALK_TASK_ID, num_envs=1, seed=1)
    conservative = make_env_cfg(CONSERVATIVE_WALK_TASK_ID, num_envs=1, seed=1)
    unloading = make_env_cfg(UNLOADING_WALK_TASK_ID, num_envs=1, seed=1)

    assert animated.observations["actor"].terms["gait_clock"].params["period"] == 0.33
    assert conservative.observations["actor"].terms["gait_clock"].params["period"] == 0.33
    assert animated.rewards["animated_reference"].params["duty_factor"] == 0.70
    assert animated.rewards["shallow_clearance"].params["target_height"] == 0.006
    assert conservative.rewards["conservative_reference"].params["duty_factor"] == 0.82
    assert conservative.rewards["shallow_clearance"].params["target_height"] == 0.003
    assert conservative.rewards["bilateral_contact"].weight == 3.0
    for cfg in (animated, conservative):
        assert cfg.rewards["pitch_upright"].weight > cfg.rewards["upright"].weight
        assert cfg.rewards["uncommanded_base_ang_vel"].func.__name__ == "pitch_dominant_angular_velocity"
        assert cfg.rewards["fixed_heading_error"].weight < 0.0
        assert cfg.rewards["lateral_path_deviation"].weight < 0.0
        assert cfg.rewards["gait_symmetry"].weight == 0.30
    assert "alternating_contact_success" in animated.metrics
    assert "bilateral_contact_success" in conservative.metrics
    assert unloading.observations["actor"].terms["gait_clock"].params["period"] == 0.33
    assert unloading.rewards["unloading_reference"].weight == 7.0
    assert unloading.rewards["scheduled_unloading"].params["stance_load_fraction"] == 0.75
    assert unloading.rewards["scheduled_unloading"].weight == 5.0
    assert "unloading_success" in unloading.metrics


def test_periodic_walk_uses_phase_composed_contact_costs_and_world_progress() -> None:
    periodic = make_env_cfg(PERIODIC_WALK_TASK_ID, num_envs=1, seed=1)
    assert periodic.observations["actor"].terms["gait_clock"].params["period"] == 0.33
    assert periodic.rewards["swing_foot_force"].weight < 0.0
    assert periodic.rewards["stance_foot_speed"].weight < 0.0
    assert periodic.rewards["world_velocity"].weight > 0.0
    assert periodic.rewards["world_progress"].weight > 0.0
    assert "animated_reference" not in periodic.rewards
    assert "conservative_reference" not in periodic.rewards
    assert "unloading_reference" not in periodic.rewards
    assert "periodic_walk_success" in periodic.metrics


def test_symmetric_periodic_walk_uses_cycle_level_straightness() -> None:
    cfg = make_env_cfg(SYMMETRIC_PERIODIC_WALK_TASK_ID, num_envs=1, seed=1)
    periodic = make_env_cfg(PERIODIC_WALK_TASK_ID, num_envs=1, seed=1)
    assert cfg.events["domain_randomization"].func.__name__ == "symmetric_domain_randomization"
    assert cfg.rewards["world_velocity"].func.__name__ == "CycleGatedWorldForwardVelocity"
    assert cfg.rewards["world_progress"].func.__name__ == "CycleGatedEpisodeWorldProgress"
    assert cfg.rewards["foot_exchange"].weight == 0.10
    assert cfg.rewards["swing_foot_force"].weight == periodic.rewards["swing_foot_force"].weight
    assert cfg.rewards["stance_foot_speed"].weight == periodic.rewards["stance_foot_speed"].weight
    assert cfg.rewards["cycle_lateral_drift"].weight == -0.10
    assert "world_lateral_velocity" not in cfg.rewards
    assert "zero_yaw_rate" in cfg.rewards
    assert "cycle_straightness_gate" in cfg.metrics
    assert "cycle_lateral_velocity_cost" in cfg.metrics
    assert "straightness_gate" not in cfg.metrics
    assert cfg.rewards["fixed_heading_error"].func.__name__ == "UnwrappedFixedHeadingError"


def test_symmetric_cycle_periodic_walk_keeps_cycle_gate_and_adds_symmetric_training() -> None:
    cfg = make_env_cfg(SYMMETRIC_CYCLE_PERIODIC_WALK_TASK_ID, num_envs=1, seed=1)
    straight = make_env_cfg(STRAIGHT_PERIODIC_WALK_TASK_ID, num_envs=1, seed=1)
    assert cfg.events["domain_randomization"].func.__name__ == "symmetric_domain_randomization"
    assert cfg.rewards["world_velocity"].func.__name__ == "CycleGatedWorldForwardVelocity"
    assert cfg.rewards["world_progress"].func.__name__ == "CycleGatedEpisodeWorldProgress"
    assert "cycle_straightness_gate" in cfg.metrics
    assert set(cfg.rewards) == set(straight.rewards)


def test_updated_default_walking_is_standalone_and_long_horizon() -> None:
    cfg = make_env_cfg(UPDATED_DEFAULT_WALKING_TASK_ID, num_envs=1, seed=1)
    periodic = make_env_cfg(PERIODIC_WALK_TASK_ID, num_envs=1, seed=1)
    assert cfg.episode_length_s == 60.0
    assert set(cfg.observations["actor"].terms) == set(periodic.observations["actor"].terms)
    assert cfg.rewards["world_velocity"].func.__name__ == "world_forward_velocity_only"
    assert cfg.rewards["cycle_lateral_drift"].func.__name__ == "UpdatedDefaultCycleLateralVelocityCost"
    assert cfg.rewards["cycle_yaw_drift"].func.__name__ == "UpdatedDefaultCycleYawRateCost"
    assert cfg.rewards["lateral_path_drift"].func.__name__ == "updated_default_lateral_path_cost"
    assert cfg.rewards["lateral_path_drift"].params["ramp_steps"] == 12_000
    assert cfg.rewards["scheduled_contact"].weight > 0.0
    assert "world_progress" not in cfg.rewards
    assert "fixed_heading_error" not in cfg.rewards
    assert "lateral_path_deviation" not in cfg.rewards
    assert cfg.events["domain_randomization"].func.__name__ == "updated_default_domain_randomization"
    assert cfg.events["domain_randomization"].params["symmetric_steps"] == 12_000
    assert "long_horizon_heading_error" in cfg.metrics
    assert "peak_lateral_deviation" in cfg.metrics
    assert "mean_lateral_deviation" in cfg.metrics
    assert "cycle_straightness_gate" in cfg.metrics


def test_updated_default_walking_velocity_range_curriculum() -> None:
    cfg = make_env_cfg(UPDATED_DEFAULT_WALKING_TASK_ID, num_envs=1, seed=1)
    assert cfg.commands["twist"].ranges.lin_vel_x == (0.075, 0.085)
    assert cfg.curriculum["velocity_command_range"].params == {
        "initial_range": (0.075, 0.085),
        "final_range": (0.0, 0.1),
        "ramp_steps": 12_000,
    }

    command_cfg = SimpleNamespace(ranges=SimpleNamespace(lin_vel_x=None))
    env = SimpleNamespace(
        common_step_counter=6_000,
        command_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(cfg=command_cfg)
        ),
    )
    state = velocity_command_range_curriculum(
        env,
        None,
        initial_range=(0.075, 0.085),
        final_range=(0.0, 0.1),
        ramp_steps=12_000,
    )
    assert command_cfg.ranges.lin_vel_x == (0.0375, 0.0925)
    assert state == {"min": 0.0375, "max": 0.0925}


def test_updated_default_lateral_walking_starts_from_completed_forward_range() -> None:
    cfg = make_env_cfg(
        UPDATED_DEFAULT_LATERAL_WALKING_TASK_ID, num_envs=1, seed=1
    )
    command = cfg.commands["twist"]
    assert command.resampling_time_range == (4.0, 8.0)
    assert command.ranges.lin_vel_x == (0.0, 0.1)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.zero_lateral_probability == 0.35
    assert cfg.curriculum["lateral_velocity_command_range"].params == {
        "initial_range": (0.0, 0.0),
        "final_range": (-0.1, 0.1),
        "ramp_steps": 24_000,
    }
    assert cfg.rewards["world_velocity"].func.__name__ == "world_forward_velocity_only"
    assert cfg.rewards["cycle_planar_velocity_error"].func.__name__ == (
        "CommandedCyclePlanarVelocityCost"
    )
    assert cfg.rewards["commanded_path_error"].func.__name__ == (
        "CommandedPlanarPathCost"
    )
    heading = cfg.rewards["heading_drift_prevention"]
    assert heading.func.__name__ == "GradualBoundedHeadingDriftCost"
    assert heading.weight == -0.05
    assert heading.params == {
        "deadband": 0.075,
        "scale": 0.25,
        "max_cost": 4.0,
        "ramp_steps": 12_000,
    }
    assert "heading_drift_prevention_cost" in cfg.metrics
    assert cfg.rewards["foot_exchange"].weight == 0.50
    assert "cycle_command_tracking_gate" in cfg.metrics
    assert cfg.events["domain_randomization"].func.__name__ == (
        "full_domain_randomization"
    )

    command_cfg = SimpleNamespace(ranges=SimpleNamespace(lin_vel_y=None))
    env = SimpleNamespace(
        common_step_counter=12_000,
        command_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(cfg=command_cfg)
        ),
    )
    state = lateral_velocity_command_range_curriculum(env, None)
    assert command_cfg.ranges.lin_vel_y == (-0.05, 0.05)
    assert state == {"min": -0.05, "max": 0.05}


def test_actor_only_same_task_initialization_preserves_curriculum_step(tmp_path) -> None:
    actor = torch.nn.Linear(2, 1)
    runner = object.__new__(TaskStampedOnPolicyRunner)
    runner.initialize_actor_only = True
    runner.task_id = UPDATED_DEFAULT_LATERAL_WALKING_TASK_ID
    runner.alg = SimpleNamespace(_raw_actor=actor)
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=0))
    checkpoint = tmp_path / "same_task.pt"
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "task_id": UPDATED_DEFAULT_LATERAL_WALKING_TASK_ID,
            "infos": {"env_state": {"common_step_counter": 24_000}},
        },
        checkpoint,
    )

    runner.load(str(checkpoint))

    assert runner.env.unwrapped.common_step_counter == 24_000


def test_actor_only_cross_task_initialization_resets_curriculum_step(tmp_path) -> None:
    actor = torch.nn.Linear(2, 1)
    runner = object.__new__(TaskStampedOnPolicyRunner)
    runner.initialize_actor_only = True
    runner.task_id = UPDATED_DEFAULT_LATERAL_WALKING_TASK_ID
    runner.alg = SimpleNamespace(_raw_actor=actor)
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=0))
    checkpoint = tmp_path / "other_task.pt"
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "task_id": UPDATED_DEFAULT_WALKING_TASK_ID,
            "infos": {"env_state": {"common_step_counter": 12_000}},
        },
        checkpoint,
    )

    runner.load(str(checkpoint))

    assert runner.env.unwrapped.common_step_counter == 0


def test_quasi_omnidirectional_task_starts_blank_with_primary_cycle_tracking() -> None:
    cfg = make_env_cfg(
        QUASI_OMNIDIRECTIONAL_WALKING_TASK_ID, num_envs=1, seed=1
    )
    command = cfg.commands["twist"]
    assert command.ranges.lin_vel_x == (0.0, 0.0)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert command.resampling_time_range == (4.0, 8.0)
    assert cfg.rewards["cycle_command_displacement"].weight == -2.0
    assert cfg.rewards["world_velocity"].weight == -0.25
    assert cfg.rewards["foot_exchange"].func.__name__ == (
        "commanded_periodic_foot_exchange"
    )
    assert "cycle_lateral_drift" not in cfg.rewards
    assert "cycle_yaw_drift" not in cfg.rewards
    assert "lateral_path_drift" not in cfg.rewards
    curriculum = cfg.curriculum["quasi_omnidirectional_command_range"].params
    assert curriculum["final_linear_speed"] == 0.1
    assert curriculum["final_yaw_rate"] == pytest.approx(torch.pi / 4.0)
    assert curriculum["ramp_steps"] == 36_000


def test_quasi_omnidirectional_v2_registers_deployment_weighted_modes() -> None:
    cfg = make_env_cfg(
        QUASI_OMNIDIRECTIONAL_WALKING_V2_TASK_ID, num_envs=1, seed=1
    )
    command = cfg.commands["twist"]
    assert isinstance(command, QuasiOmnidirectionalMixtureCommandCfg)
    probabilities = (
        command.forward_straight_probability,
        command.forward_curve_probability,
        command.pure_lateral_probability,
        command.full_omni_probability,
        command.stop_probability,
        command.reverse_probability,
    )
    assert probabilities == pytest.approx((0.30, 0.25, 0.15, 0.20, 0.05, 0.05))
    assert sum(probabilities) == pytest.approx(1.0)

    # V2 combines short-horizon gait-cycle tracking with an episode-long
    # displacement term.  Both are intentionally the largest locomotion
    # penalty weights so neither local nor persistent under-travel is ignored.
    assert "episode_command_displacement" in cfg.rewards
    assert cfg.rewards["cycle_command_displacement"].weight == -3.5
    assert cfg.rewards["episode_command_displacement"].weight == -8.0
    assert "episode_command_displacement_cost" in cfg.metrics
    assert {
        name: term.weight for name, term in cfg.rewards.items()
    } == QUASI_OMNIDIRECTIONAL_V2_REWARD_WEIGHTS
    assert cfg.rewards["actuator_effort"].params["joint_weights"] == (
        QUASI_OMNIDIRECTIONAL_V2_ACTUATOR_EFFORT_JOINT_WEIGHTS
    )

    locomotion_penalties = {
        name: term.weight
        for name, term in cfg.rewards.items()
        if term.weight < 0.0 and name != "termination"
    }
    top_two = {
        name
        for name, _ in sorted(
            locomotion_penalties.items(), key=lambda item: abs(item[1]), reverse=True
        )[:2]
    }
    assert top_two == {
        "cycle_command_displacement",
        "episode_command_displacement",
    }


def test_quasi_omnidirectional_v2_forced_modes_have_expected_axes() -> None:
    def make_command(mode: int) -> QuasiOmnidirectionalMixtureCommand:
        probabilities = [0.0] * 6
        probabilities[mode] = 1.0
        command = object.__new__(QuasiOmnidirectionalMixtureCommand)
        command.cfg = QuasiOmnidirectionalMixtureCommandCfg(
            entity_name="robot",
            resampling_time_range=(4.0, 8.0),
            heading_command=False,
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            rel_world_envs=0.0,
            rel_forward_envs=0.0,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.1, 0.1),
                lin_vel_y=(-0.1, 0.1),
                ang_vel_z=(-torch.pi / 4.0, torch.pi / 4.0),
                heading=None,
            ),
            forward_straight_probability=probabilities[0],
            forward_curve_probability=probabilities[1],
            pure_lateral_probability=probabilities[2],
            full_omni_probability=probabilities[3],
            stop_probability=probabilities[4],
            reverse_probability=probabilities[5],
        )
        command._env = SimpleNamespace(device="cpu")
        count = 256
        command.vel_command_b = torch.zeros(count, 3)
        command.vel_command_w = torch.zeros(count, 3)
        command.is_heading_env = torch.zeros(count, dtype=torch.bool)
        command.is_standing_env = torch.zeros(count, dtype=torch.bool)
        command.is_world_env = torch.zeros(count, dtype=torch.bool)
        command.is_forward_env = torch.zeros(count, dtype=torch.bool)
        command._resample_command(torch.arange(count))
        return command

    straight = make_command(QuasiOmnidirectionalMixtureCommand._FORWARD_STRAIGHT)
    assert torch.all(straight.vel_command_b[:, 0] >= 0.0)
    assert torch.all(straight.vel_command_b[:, 1:] == 0.0)

    curve = make_command(QuasiOmnidirectionalMixtureCommand._FORWARD_CURVE)
    assert torch.all(curve.vel_command_b[:, 0] >= 0.0)
    assert torch.all(curve.vel_command_b[:, 1] == 0.0)
    assert torch.all(torch.abs(curve.vel_command_b[:, 2]) <= torch.pi / 4.0)

    lateral = make_command(QuasiOmnidirectionalMixtureCommand._PURE_LATERAL)
    assert torch.all(lateral.vel_command_b[:, 0] == 0.0)
    assert torch.all(lateral.vel_command_b[:, 2] == 0.0)

    omni = make_command(QuasiOmnidirectionalMixtureCommand._FULL_OMNI)
    assert torch.all(torch.abs(omni.vel_command_b[:, 0]) <= 0.1)
    assert torch.all(torch.abs(omni.vel_command_b[:, 1]) <= 0.1)
    assert torch.all(torch.abs(omni.vel_command_b[:, 2]) <= torch.pi / 4.0)

    stop = make_command(QuasiOmnidirectionalMixtureCommand._STOP)
    assert torch.all(stop.vel_command_b == 0.0)

    reverse = make_command(QuasiOmnidirectionalMixtureCommand._REVERSE)
    assert torch.all(reverse.vel_command_b[:, 0] <= 0.0)
    assert torch.all(reverse.vel_command_b[:, 1:] == 0.0)


def test_periodic_reward_composition_sampler_never_combines_planar_axes() -> None:
    cfg = make_env_cfg(
        PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID, num_envs=1, seed=1
    ).commands["twist"]
    command = object.__new__(QuasiOmnidirectionalMixtureCommand)
    command.cfg = cfg
    command._env = SimpleNamespace(device="cpu")
    count = 4096
    command.vel_command_b = torch.zeros(count, 3)
    command.vel_command_w = torch.zeros(count, 3)
    command.is_heading_env = torch.zeros(count, dtype=torch.bool)
    command.is_standing_env = torch.zeros(count, dtype=torch.bool)
    command.is_world_env = torch.zeros(count, dtype=torch.bool)
    command.is_forward_env = torch.zeros(count, dtype=torch.bool)
    command._resample_command(torch.arange(count))

    moving_x = command.vel_command_b[:, 0] != 0.0
    moving_y = command.vel_command_b[:, 1] != 0.0
    assert not torch.any(moving_x & moving_y)
    assert torch.all(command.vel_command_b[:, 2] == 0.0)
    assert moving_x.any() and moving_y.any() and (~moving_x & ~moving_y).any()


def test_natural_v3_sampler_has_only_exclusive_axes_or_forward_yaw() -> None:
    cfg = make_env_cfg(
        NATURAL_OMNIDIRECTIONAL_V3_TASK_ID, num_envs=1, seed=1
    ).commands["twist"]
    command = object.__new__(QuasiOmnidirectionalMixtureCommand)
    command.cfg = cfg
    command._env = SimpleNamespace(device="cpu")
    count = 4096
    command.vel_command_b = torch.zeros(count, 3)
    command.vel_command_w = torch.zeros(count, 3)
    command.is_heading_env = torch.zeros(count, dtype=torch.bool)
    command.is_standing_env = torch.zeros(count, dtype=torch.bool)
    command.is_world_env = torch.zeros(count, dtype=torch.bool)
    command.is_forward_env = torch.zeros(count, dtype=torch.bool)
    command._resample_command(torch.arange(count))

    values = command.vel_command_b
    moving_x = values[:, 0].abs() > 1e-6
    moving_y = values[:, 1].abs() > 1e-6
    turning = values[:, 2].abs() > 1e-6
    assert not torch.any(moving_x & moving_y)
    assert not torch.any(turning & ~moving_x)
    assert torch.all(values[:, 0] >= -.05)
    assert torch.all(values[:, 0] <= .10)
    assert torch.all(values[:, 1].abs() <= .08)
    assert torch.all(values[:, 2].abs() <= .40)
    # V3 deliberately excludes backward motion; the current deployment modes
    # are forward, curved-forward, lateral, and stopped.
    assert torch.all(values[:, 0] >= 0.0)
    assert torch.any(turning)
    assert torch.any(~moving_x & ~moving_y & ~turning)


def test_quasi_omnidirectional_training_and_playback_keep_action_authority() -> None:
    training = make_env_cfg(
        QUASI_OMNIDIRECTIONAL_WALKING_TASK_ID, num_envs=1, seed=1, play=False
    )
    playback = make_env_cfg(
        QUASI_OMNIDIRECTIONAL_WALKING_TASK_ID, num_envs=1, seed=1, play=True
    )
    assert training.actions["joint_pos"].initial_range_fraction == 1.0
    assert playback.actions["joint_pos"].initial_range_fraction == 1.0
    assert training.actions["joint_pos"].range_ramp_steps == 0
    assert playback.actions["joint_pos"].range_ramp_steps == 0


def test_quasi_omnidirectional_curriculum_expands_all_command_axes() -> None:
    command_cfg = SimpleNamespace(
        ranges=SimpleNamespace(lin_vel_x=None, lin_vel_y=None, ang_vel_z=None)
    )
    env = SimpleNamespace(
        common_step_counter=18_000,
        command_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(cfg=command_cfg)
        ),
    )

    state = quasi_omnidirectional_command_curriculum(env, None)

    assert command_cfg.ranges.lin_vel_x == pytest.approx((-0.05, 0.05))
    assert command_cfg.ranges.lin_vel_y == pytest.approx((-0.05, 0.05))
    assert command_cfg.ranges.ang_vel_z == pytest.approx(
        (-torch.pi / 8.0, torch.pi / 8.0)
    )
    assert state["yaw_max"] == pytest.approx(torch.pi / 8.0)


def test_staged_omni_curriculum_protects_forward_and_advances_after_gates() -> None:
    command_cfg = SimpleNamespace(
        ranges=SimpleNamespace(lin_vel_x=None, lin_vel_y=None, ang_vel_z=None),
        protected_forward_range=None,
        forward_straight_probability=0.3,
        forward_curve_probability=0.2,
        pure_lateral_probability=0.1,
        full_omni_probability=0.1,
        stop_probability=0.1,
        reverse_probability=0.2,
    )
    metrics = SimpleNamespace(
        _step_count=torch.zeros(2, dtype=torch.long),
        _episode_sums={
            name: torch.zeros(2)
            for name in (
                "command_tracking_success",
                "cycle_command_displacement_cost",
                "episode_command_displacement_cost",
            )
        },
    )
    termination = SimpleNamespace(
        active_terms=["time_out", "fell_over", "illegal_contact"],
        get_term=lambda name: torch.tensor([name == "time_out", name == "time_out"]),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        command_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(cfg=command_cfg)
        ),
        metrics_manager=metrics,
        termination_manager=termination,
    )
    curriculum = StagedQuasiOmnidirectionalCurriculum(
        window_episodes=2, advance_windows=1
    )
    curriculum(env, None)
    assert command_cfg.ranges.lin_vel_x == pytest.approx((0.04, 0.10))
    assert command_cfg.ranges.lin_vel_y == pytest.approx((0.0, 0.0))

    metrics._step_count[:] = 300
    metrics._episode_sums["command_tracking_success"][:] = 300
    metrics._episode_sums["cycle_command_displacement_cost"][:] = 100
    metrics._episode_sums["episode_command_displacement_cost"][:] = 100
    state = curriculum(env, None)
    assert state["stage"] == 1.0
    assert command_cfg.ranges.lin_vel_x == pytest.approx((0.04, 0.10))
    assert command_cfg.ranges.ang_vel_z == pytest.approx((-0.05, 0.05))


def test_quasi_omnidirectional_yaw_is_bounded_by_forward_command() -> None:
    command = object.__new__(QuasiOmnidirectionalCommand)
    command.cfg = QuasiOmnidirectionalCommandCfg(
        entity_name="robot",
        resampling_time_range=(4.0, 8.0),
        heading_command=False,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        zero_forward_probability=0.0,
        zero_lateral_probability=0.0,
        zero_yaw_probability=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-torch.pi / 4.0, torch.pi / 4.0),
            heading=None,
        ),
    )
    command._env = SimpleNamespace(device="cpu")
    count = 4096
    command.vel_command_b = torch.zeros(count, 3)
    command.vel_command_w = torch.zeros(count, 3)
    command.is_heading_env = torch.zeros(count, dtype=torch.bool)
    command.is_standing_env = torch.zeros(count, dtype=torch.bool)
    command.is_world_env = torch.zeros(count, dtype=torch.bool)
    command.is_forward_env = torch.zeros(count, dtype=torch.bool)

    command._resample_command(torch.arange(count))

    allowed = (torch.pi / 4.0) * torch.abs(command.vel_command_b[:, 0]) / 0.1
    assert torch.all(torch.abs(command.vel_command_b[:, 2]) <= allowed + 1.0e-6)


def test_straight_periodic_walk_is_the_minimal_lateral_motion_correction() -> None:
    cfg = make_env_cfg(STRAIGHT_PERIODIC_WALK_TASK_ID, num_envs=1, seed=1)
    periodic = make_env_cfg(PERIODIC_WALK_TASK_ID, num_envs=1, seed=1)
    assert cfg.events["domain_randomization"].func is periodic.events["domain_randomization"].func
    assert cfg.rewards["world_velocity"].func.__name__ == "CycleGatedWorldForwardVelocity"
    assert cfg.rewards["world_progress"].func.__name__ == "CycleGatedEpisodeWorldProgress"
    assert cfg.rewards["cycle_lateral_drift"].weight < 0.0
    assert "world_lateral_velocity" not in cfg.rewards
    assert cfg.rewards["foot_exchange"].weight == periodic.rewards["foot_exchange"].weight
    assert "cycle_straightness_gate" in cfg.metrics
    assert "cycle_lateral_velocity_cost" in cfg.metrics
    assert cfg.rewards["fixed_heading_error"].func.__name__ == "UnwrappedFixedHeadingError"


def test_weight_transfer_reward_matches_force_based_promotion_metric() -> None:
    cfg = make_env_cfg(WEIGHT_TRANSFER_TASK_ID, num_envs=1, seed=1)
    reward = cfg.rewards["load_transfer"]
    metric = cfg.metrics["transfer_success"]
    assert reward.func is metric.func
    assert reward.weight == 10.0
    assert cfg.rewards["both_feet_contact"].weight == 1.0
    assert cfg.rewards["weight_transfer"].params["amplitude"] == 0.14
    assert cfg.rewards["weight_transfer"].params["initial_amplitude"] == 0.0
    assert cfg.rewards["weight_transfer"].params["ramp_steps"] == 12_000
    assert reward.params["target_imbalance"] == 0.60
    assert reward.params["std"] == 0.12
    assert reward.params["initial_imbalance"] == 0.0
    assert reward.params["ramp_steps"] == 12_000


def test_skill_tasks_have_ppo_configs() -> None:
    register_tasks()
    assert all(
        load_rl_cfg(task).actor.distribution_cfg["init_std"] == 0.10
        for task in LOCOMOTION_TASK_IDS
        if task not in (
            NATURAL_OMNIDIRECTIONAL_TASK_ID,
            NATURAL_OMNIDIRECTIONAL_V3_TASK_ID,
            NATURAL_OMNIDIRECTIONAL_V4_TASK_ID,
            NATURAL_OMNIDIRECTIONAL_V5_TASK_ID,
            PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID,
        )
    )
    assert load_rl_cfg(NATURAL_OMNIDIRECTIONAL_TASK_ID).actor.distribution_cfg["init_std"] == 0.15
    assert load_rl_cfg(NATURAL_OMNIDIRECTIONAL_V3_TASK_ID).actor.distribution_cfg["init_std"] == 0.15
    assert load_rl_cfg(NATURAL_OMNIDIRECTIONAL_V4_TASK_ID).actor.distribution_cfg["init_std"] == 0.15
    assert load_rl_cfg(NATURAL_OMNIDIRECTIONAL_V5_TASK_ID).actor.distribution_cfg["init_std"] == 0.15
    assert load_rl_cfg(PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID).actor.distribution_cfg["init_std"] == 0.12
    symmetry_cfg = load_rl_cfg(SYMMETRIC_PERIODIC_WALK_TASK_ID).algorithm.symmetry_cfg
    assert symmetry_cfg is not None
    assert symmetry_cfg["use_data_augmentation"] is True
    assert symmetry_cfg["use_mirror_loss"] is False
    cycle_symmetry_cfg = load_rl_cfg(
        SYMMETRIC_CYCLE_PERIODIC_WALK_TASK_ID
    ).algorithm.symmetry_cfg
    assert cycle_symmetry_cfg is not None
    assert cycle_symmetry_cfg["use_data_augmentation"] is True
    updated_cfg = load_rl_cfg(UPDATED_DEFAULT_WALKING_TASK_ID).algorithm.symmetry_cfg
    assert updated_cfg is not None
    assert updated_cfg["use_data_augmentation"] is True
    lateral_cfg = load_rl_cfg(
        UPDATED_DEFAULT_LATERAL_WALKING_TASK_ID
    ).algorithm.symmetry_cfg
    assert lateral_cfg is not None
    assert lateral_cfg["use_data_augmentation"] is True


def test_sagittal_task_uses_nonzero_discrete_forward_and_backward_bins() -> None:
    cfg = make_env_cfg(SAGITTAL_TASK_ID, num_envs=1, seed=1)
    assert isinstance(cfg.commands["twist"], DiscreteSagittalCommandCfg)
    assert cfg.commands["twist"].bins == (-0.50, -0.30, -0.15, 0.15, 0.30, 0.50)


def test_skill_checkpoints_are_deployable_by_fixed_ankle_profile() -> None:
    assert all(LEGGYBOT_FIXED_ANKLE_PROFILE.supports_task(task) for task in LOCOMOTION_TASK_IDS)
    assert all(
        LEGGYBOT_FIXED_ANKLE_PROFILE.observation_dim_for_task(task) == 29
        for task in LOCOMOTION_TASK_IDS
    )


def test_weight_transfer_metric_uses_vertical_load_distribution() -> None:
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            found=torch.tensor([[1, 1], [1, 1]]),
            force=torch.tensor([
                [[5.0, 0.0, 10.0], [5.0, 0.0, 10.0]],
                [[5.0, 0.0, 16.0], [5.0, 0.0, 4.0]],
            ]),
        )
    )
    env = SimpleNamespace(
        scene={"feet_ground_contact": sensor},
        episode_length_buf=torch.tensor([0, 50]),
        step_dt=0.01,
    )
    score = lateral_support_balance(env, "feet_ground_contact")
    # At phase 0 the target is balanced; at phase 0.25 the target is +60%
    # right-minus-left load, exactly matching 16/(16+4).
    torch.testing.assert_close(score, torch.ones(2), atol=1e-6, rtol=0)
