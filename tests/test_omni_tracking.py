from types import SimpleNamespace

import torch

from locomotion.tasks.omni_tracking import WindowedSE2Tracking, integrate_twist


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
