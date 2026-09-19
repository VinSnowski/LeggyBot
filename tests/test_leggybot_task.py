from types import SimpleNamespace

import torch

from locomotion.tasks.gait import (
    actuator_effort_cost,
    commanded_lateral_foot_line_cost,
    commanded_motion_rate_safety_cost,
    gait_clock,
    lateral_foot_line_error,
    pitch_dominant_angular_velocity,
    pitch_upright,
    sim2real_randomization_scale,
    updated_default_domain_randomization,
)


def test_randomization_ramps_from_paired_to_full_domain(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "locomotion.tasks.gait.symmetric_domain_randomization",
        lambda env, ids: calls.append(("symmetric", float(env.common_step_counter))),
    )
    monkeypatch.setattr(
        "locomotion.tasks.gait.full_domain_randomization",
        lambda env, ids, **kwargs: calls.append(
            ("full", float(env.common_step_counter), len(ids))
        ),
    )
    monkeypatch.setattr(
        "locomotion.tasks.gait._interpolated_full_domain_randomization",
        lambda env, ids, amplitude: calls.append(("ramp", float(amplitude))),
    )
    env = SimpleNamespace(common_step_counter=12_000, device="cpu")
    ids = torch.arange(2)

    updated_default_domain_randomization(
        env, ids, symmetric_steps=12_000, independent_ramp_steps=12_000
    )
    assert calls == [("symmetric", 12000.0)]

    calls.clear()
    env.common_step_counter = 18_000
    monkeypatch.setattr(
        "locomotion.tasks.gait.torch.rand",
        lambda shape, device: torch.tensor([0.25, 0.75], device=device),
    )
    updated_default_domain_randomization(
        env, ids, symmetric_steps=12_000, independent_ramp_steps=12_000
    )
    assert calls == [("ramp", 0.5)]

    calls.clear()
    env.common_step_counter = 24_000
    updated_default_domain_randomization(
        env, ids, symmetric_steps=12_000, independent_ramp_steps=12_000
    )
    assert calls == [("full", 24000.0, 2)]


def test_gait_clock_has_a_half_cycle_offset() -> None:
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([0, 8, 16]), step_dt=0.01
    )
    clock = gait_clock(env, period=0.32)
    torch.testing.assert_close(clock[0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(
        clock[2], torch.tensor([0.0, -1.0]), atol=1e-6, rtol=0
    )


def test_commanded_motion_rate_safety_uses_body_lateral_velocity() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_w=torch.tensor([[0.0, 1.0, 0.0]]),
            root_link_lin_vel_b=torch.tensor([[0.0, 0.1, 0.0]]),
            root_link_ang_vel_b=torch.zeros(1, 3),
        )
    )
    command = torch.tensor([[0.0, 0.1, 0.0]])
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda _: command),
    )
    torch.testing.assert_close(
        commanded_motion_rate_safety_cost(env, "twist"), torch.zeros(1)
    )


def test_lateral_foot_line_term_is_gated_by_command_and_support() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            site_pos_w=torch.tensor(
                [[[0.020, 0.040, 0.0], [0.000, -0.040, 0.0]]]
            ),
            heading_w=torch.zeros(1),
        )
    )
    command = torch.tensor([[0.0, 0.08, 0.0]])
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([0]),
        step_dt=0.02,
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda _: command),
    )
    site_cfg = SimpleNamespace(name="robot", site_ids=[0, 1])

    torch.testing.assert_close(
        lateral_foot_line_error(env, "twist", asset_cfg=site_cfg),
        torch.tensor([0.020]),
    )
    assert commanded_lateral_foot_line_cost(
        env, "twist", line_scale=0.015, asset_cfg=site_cfg
    ).item() > 0.0

    command[:] = torch.tensor([[0.08, 0.0, 0.0]])
    torch.testing.assert_close(
        commanded_lateral_foot_line_cost(env, "twist", asset_cfg=site_cfg),
        torch.zeros(1),
    )


def test_actuator_effort_supports_per_joint_weights() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(qfrc_actuator=torch.ones(1, 6))
    )
    env = SimpleNamespace(scene={"robot": robot})
    baseline = actuator_effort_cost(env, effort_scale=1.5)
    weighted = actuator_effort_cost(
        env,
        effort_scale=1.5,
        joint_weights=(1.0, 0.25, 0.25, 1.0, 0.25, 0.25),
    )
    assert weighted.item() < baseline.item()


def test_pitch_terms_prioritize_pitch_over_roll_and_yaw() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            projected_gravity_b=torch.tensor(
                [[0.0, 0.0, -1.0], [0.2, 0.0, -0.98]]
            ),
            root_link_ang_vel_b=torch.tensor(
                [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
            ),
        )
    )
    command = torch.zeros(2, 3)
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda _: command),
    )
    assert pitch_upright(env)[0] > pitch_upright(env)[1]
    penalty = pitch_dominant_angular_velocity(env, "twist")
    assert penalty[1] > penalty[0]


def test_sim2real_randomization_is_enabled_by_default() -> None:
    assert sim2real_randomization_scale(SimpleNamespace(cfg=SimpleNamespace())) == 1.0
    assert sim2real_randomization_scale(
        SimpleNamespace(cfg=SimpleNamespace(sim2real_randomization_enabled=False))
    ) == 0.0
