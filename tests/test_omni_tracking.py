from types import SimpleNamespace

import torch

from locomotion.tasks.omni_tracking import WindowedSE2Tracking, integrate_twist
from locomotion.task import make_env_cfg, register_tasks
from locomotion.tasks.skills import (
    NATURAL_OMNIDIRECTIONAL_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_V3_TASK_ID,
    PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID,
    SE2_TRACKING_TASK_ID,
    get_locomotion_task,
)
from mjlab.tasks.registry import load_rl_cfg


def fixture(command=(.08, 0., 0.), count=2, horizon=.66):
    data = SimpleNamespace(root_link_pos_w=torch.zeros(count, 3),
                           heading_w=torch.zeros(count),
                           projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(count, 1))
    commands = torch.tensor([command]).repeat(count, 1)
    env = SimpleNamespace(num_envs=count, device="cpu", step_dt=.02,
                          episode_length_buf=torch.zeros(count, dtype=torch.long),
                          scene={"robot": SimpleNamespace(data=data)},
                          command_manager=SimpleNamespace(get_command=lambda _: commands))
    term = WindowedSE2Tracking(SimpleNamespace(params={"horizon_s": horizon}), env)
    return env, data, commands, term


def advance(env, data, pose):
    data.root_link_pos_w[:, :2] = pose[:, :2]
    data.heading_w[:] = torch.atan2(pose[:, 2].sin(), pose[:, 2].cos())
    env.episode_length_buf += 1


def test_exact_se2_arc_and_zero_yaw():
    pose = torch.zeros(2, 3)
    commands = torch.tensor([[1., 0., 0.], [1., 0., 1.]])
    result = integrate_twist(pose, commands, torch.pi/2)
    torch.testing.assert_close(result, torch.tensor([[torch.pi/2, 0., 0.], [1., 1., torch.pi/2]]))


def test_perfect_curves_and_command_changes_track_through_heading_wrap():
    env, data, commands, term = fixture((.08, .03, .4))
    pose = torch.zeros(2, 3)
    term(env)
    for i in range(500):
        if i == 100:
            commands[:, 1] *= -1
        pose = integrate_twist(pose, commands, env.step_dt)
        advance(env, data, pose)
        torch.testing.assert_close(term.errors(env), torch.zeros(2, 3), atol=2e-5, rtol=0)


def test_straight_yaw_drift_and_stationary_undertravel_are_penalized():
    env, data, commands, term = fixture()
    pose = torch.zeros(2, 3)
    actual = commands.clone()
    actual[0, 2] = .3  # Correct forward speed, unwanted curved path.
    actual[1] = 0.    # Standing is not successful forward walking.
    term(env)
    for _ in range(100):
        pose = integrate_twist(pose, actual, env.step_dt)
        advance(env, data, pose)
        term(env)
    error = term.errors(env)
    assert error[0, 2] > .29
    assert error[0, 1] > .005
    assert error[1, 0] < -.079
    assert not term(env, output="success").any()


def test_history_is_local_idempotent_and_partial_reset_safe():
    env, data, commands, term = fixture()
    pose = torch.zeros(2, 3)
    term(env)
    for i in range(100):
        # A bad initial segment must age out of the reward.
        pose = integrate_twist(pose, commands if i > 20 else commands*0, env.step_dt)
        advance(env, data, pose)
        before = term(env).clone()
        age = term.age.clone()
        torch.testing.assert_close(term(env), before)
        torch.testing.assert_close(term.age, age)
    torch.testing.assert_close(term.errors(env), torch.zeros(2, 3), atol=1e-5, rtol=0)
    term.reset(torch.tensor([0]))
    env.episode_length_buf[0] = 0
    data.root_link_pos_w[0] = 100.
    term(env)
    assert term.age.tolist() == [0, 100]
    torch.testing.assert_close(term.error, torch.zeros(2, 3), atol=1e-5, rtol=0)


def test_displacement_outputs_accumulate_persistent_bias_without_time_division():
    env, data, commands, term = fixture((.08, 0., .30), count=1, horizon=4.)
    pose = torch.zeros(1, 3)
    actual = commands.clone()
    actual[:, 2] -= .05
    term(env)
    for _ in range(200):
        pose = integrate_twist(pose, actual, env.step_dt)
        advance(env, data, pose)
        term(env)

    torch.testing.assert_close(
        term(env, output="yaw_error"), torch.tensor([.05]), atol=2e-5, rtol=0
    )
    torch.testing.assert_close(
        term(env, output="yaw_displacement_error"),
        torch.tensor([.20]), atol=2e-5, rtol=0,
    )
    expected = torch.sqrt(torch.tensor(1.0 + (.20 / .12) ** 2)) - 1.0
    torch.testing.assert_close(
        term(env, yaw_scale=.12, output="yaw_displacement_cost"),
        expected[None], atol=2e-5, rtol=0,
    )


def test_new_task_trains_every_required_mode_and_uses_realistic_randomization():
    task = get_locomotion_task(SE2_TRACKING_TASK_ID)
    cfg = task.environment_factory()
    command = cfg.commands["twist"]
    assert command.pure_lateral_probability > 0
    assert command.forward_curve_probability > 0
    assert command.ranges.ang_vel_z[1] > 0
    assert command.ranges.lin_vel_y[1] > 0
    assert not cfg.curriculum
    assert not task.use_symmetry_augmentation
    assert cfg.events["domain_randomization"].params["symmetric_steps"] == 0
    assert "episode_command_displacement" not in cfg.rewards


def test_natural_task_gates_gait_shape_and_bounds_ppo_exploration():
    register_tasks()
    task = get_locomotion_task(NATURAL_OMNIDIRECTIONAL_TASK_ID)
    cfg = make_env_cfg(NATURAL_OMNIDIRECTIONAL_TASK_ID, num_envs=2, seed=1)
    command = cfg.commands["twist"]
    assert command.minimum_lateral_speed == .04
    assert command.stop_probability == .10
    assert cfg.rewards["scheduled_contact"].func.__name__ == "commanded_scheduled_foot_contact"
    assert cfg.rewards["swing_foot_force"].func.__name__ == "commanded_periodic_swing_force_cost"
    assert cfg.rewards["stance_foot_speed"].func.__name__ == "commanded_periodic_stance_foot_speed_cost"
    assert {"linear_tracking", "yaw_tracking", "neutral_stance", "stopped_foot_alignment"} <= set(cfg.rewards)
    assert "se2_tracking" not in cfg.rewards
    assert not task.use_symmetry_augmentation
    runner = load_rl_cfg(NATURAL_OMNIDIRECTIONAL_TASK_ID)
    assert runner.actor.distribution_cfg["std_type"] == "log"
    assert runner.actor.distribution_cfg["std_range"] == (.03, .40)
    assert runner.algorithm.entropy_coef == .001


def test_natural_v3_uses_exclusive_modes_and_natural_ppo_settings():
    register_tasks()
    task = get_locomotion_task(NATURAL_OMNIDIRECTIONAL_V3_TASK_ID)
    cfg = make_env_cfg(NATURAL_OMNIDIRECTIONAL_V3_TASK_ID, num_envs=2, seed=1)
    command = cfg.commands["twist"]
    assert command.ranges.lin_vel_x == (-.05, .10)
    assert command.ranges.lin_vel_y == (-.08, .08)
    assert command.ranges.ang_vel_z == (-.40, .40)
    assert command.forward_straight_probability == .35
    assert command.forward_curve_probability == .35
    assert command.pure_lateral_probability == .20
    assert command.stop_probability == .10
    assert command.reverse_probability == 0.
    assert command.full_omni_probability == 0.
    assert command.resampling_time_range == (5., 8.)
    assert command.minimum_coupled_yaw_fraction == .25
    assert command.minimum_lateral_speed == .04
    assert cfg.rewards["pitch_upright"].weight == 1.50
    assert "linear_velocity_precision" not in cfg.rewards
    assert "yaw_velocity_precision" not in cfg.rewards
    assert cfg.rewards["linear_path_error"].params["horizon_s"] == 2.0
    assert cfg.rewards["yaw_path_error"].params["horizon_s"] == 2.0
    assert "lateral_foot_line" not in cfg.rewards
    assert "lateral_foot_line_error_m" in cfg.metrics
    assert {
        "left_lateral_command_fraction",
        "right_lateral_command_fraction",
        "left_lateral_tracking_success",
        "right_lateral_tracking_success",
        "left_lateral_foot_line_error_m",
        "right_lateral_foot_line_error_m",
    } <= set(cfg.metrics)
    randomization = cfg.events["domain_randomization"]
    assert randomization.func.__name__ == "updated_default_domain_randomization"
    assert randomization.params["symmetric_steps"] == 500 * 24
    assert randomization.params["independent_ramp_steps"] == 300 * 24
    assert not cfg.curriculum
    assert not task.use_symmetry_augmentation
    assert task.symmetry_mirror_loss_coeff == .10
    runner = load_rl_cfg(NATURAL_OMNIDIRECTIONAL_V3_TASK_ID)
    assert runner.actor.distribution_cfg["init_std"] == .15
    assert runner.actor.distribution_cfg["std_range"] == (.03, .40)
    assert runner.algorithm.learning_rate == 1e-3


def test_periodic_reward_composition_task_is_axis_exclusive_and_reference_free():
    register_tasks()
    task = get_locomotion_task(PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID)
    cfg = make_env_cfg(
        PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID, num_envs=2, seed=1
    )
    command = cfg.commands["twist"]
    assert command.forward_straight_probability == .45
    assert command.pure_lateral_probability == .45
    assert command.stop_probability == .10
    assert command.forward_curve_probability == 0.
    assert command.full_omni_probability == 0.
    assert command.reverse_probability == 0.
    assert command.ranges.ang_vel_z == (0., 0.)
    assert command.minimum_lateral_speed == .04
    assert "scheduled_contact" not in cfg.rewards
    assert "linear_path_error" not in cfg.rewards
    assert cfg.rewards["forward_velocity"].weight == -1.20
    assert cfg.rewards["lateral_velocity"].weight == -1.20
    assert cfg.rewards["swing_foot_force"].weight == -.40
    assert cfg.rewards["swing_foot_force"].params["force_scale"] == 10.
    assert cfg.rewards["stance_foot_speed"].weight == -.40
    assert cfg.rewards["stance_foot_speed"].params["speed_coefficient"] == 2.
    assert cfg.rewards["base_orientation"].weight == -.30
    assert cfg.rewards["alive_bias"].weight == 1.
    assert "moving_tracking_success" not in cfg.metrics
    assert not cfg.curriculum
    assert not task.use_symmetry_augmentation
    assert task.symmetry_mirror_loss_coeff == .10
    runner = load_rl_cfg(PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID)
    assert runner.actor.class_name == "MLPModel"
    assert runner.actor.distribution_cfg["init_std"] == .12
    assert runner.actor.distribution_cfg["std_range"] == (.04, .30)
    assert runner.algorithm.learning_rate == 1e-4
