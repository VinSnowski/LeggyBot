from types import SimpleNamespace

import torch
from locomotion.tasks.gait import (
    animated_walk_reference_joint_positions,
    commanded_periodic_foot_exchange,
    commanded_motion_rate_safety_cost,
    commanded_lateral_foot_line_cost,
    instantaneous_world_command_velocity_cost,
    actuator_effort_cost,
    CommandedCycleDisplacementCost,
    CommandedEpisodeDisplacementCost,
    CommandedCyclePlanarVelocityCost,
    CommandedPlanarPathCost,
    CycleNetLateralVelocityCost,
    CycleNetMotionGate,
    CycleNetYawRateCost,
    conservative_shuffle_reference_joint_positions,
    desired_stance_mask,
    gait_clock,
    gait_reference_joint_positions,
    pitch_dominant_angular_velocity,
    periodic_stance_foot_speed_cost,
    periodic_swing_force_cost,
    periodic_reward_composition_action_difference_cost,
    periodic_reward_composition_orientation_cost,
    periodic_reward_composition_velocity_cost,
    smooth_swing_probability,
    pitch_upright,
    lateral_foot_line_error,
    sim2real_randomization_scale,
    updated_default_domain_randomization,
    scheduled_swing_unloading,
    track_animated_walk_reference,
    unloading_walk_reference_joint_positions,
    upright_gate,
)
from locomotion.tasks.rewards import (
    GradualBoundedHeadingDriftCost,
    UnwrappedFixedHeadingError,
    command_direction_progress,
    fixed_heading_error,
    lateral_path_deviation,
    uncommanded_base_angular_velocity,
)


def test_updated_randomization_ramps_from_paired_to_full_domain(monkeypatch):
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
        lambda shape, device: torch.tensor([.25, .75], device=device),
    )
    updated_default_domain_randomization(
        env, ids, symmetric_steps=12_000, independent_ramp_steps=12_000
    )
    assert calls == [("ramp", .5)]

    calls.clear()
    env.common_step_counter = 24_000
    updated_default_domain_randomization(
        env, ids, symmetric_steps=12_000, independent_ramp_steps=12_000
    )
    assert calls == [("full", 24000.0, 2)]


def test_gait_clock_contact_schedule_and_reference_are_half_cycle_offset() -> None:
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([0, 6, 20, 26, 32]), step_dt=0.02
    )
    clock = gait_clock(env)
    torch.testing.assert_close(clock[0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(clock[2], torch.tensor([0.0, -1.0]), atol=1e-6, rtol=0)

    stance = desired_stance_mask(env)
    assert stance[0].tolist() == [True, True]
    assert stance[1].tolist() == [True, False]
    assert stance[3].tolist() == [False, True]

    robot = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=torch.zeros(5, 6),
            joint_pos=torch.zeros(5, 6),
        )
    )
    env.scene = {"robot": robot}
    reference = gait_reference_joint_positions(env)
    torch.testing.assert_close(reference[4, 1], torch.tensor(0.57), atol=1e-6, rtol=0)
    torch.testing.assert_close(reference[4, 2], torch.tensor(-0.995), atol=1e-6, rtol=0)
    torch.testing.assert_close(reference[4, 4], torch.tensor(0.0), atol=1e-6, rtol=0)


def test_upright_gate_removes_gait_reward_near_the_fall_limit() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            projected_gravity_b=torch.tensor(
                [[0.0, 0.0, -1.0], [0.0, 0.0, -0.7], [0.0, 0.0, -0.5]]
            )
        )
    )
    env = SimpleNamespace(scene={"robot": robot})

    torch.testing.assert_close(upright_gate(env), torch.tensor([1.0, 0.5, 0.0]))


def test_periodic_reward_composition_auxiliary_costs_match_paper_kernels() -> None:
    command = torch.tensor([[0.08, -0.04, 0.0], [0.0, 0.0, 0.0]])
    pitch = torch.tensor(torch.pi / 6)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_b=torch.tensor([[0.03, -0.01, 0.0], [0.0, 0.0, 0.0]]),
            root_link_quat_w=torch.tensor(
                [
                    [torch.cos(pitch / 2), 0.0, torch.sin(pitch / 2), 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ),
        )
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda _: command),
        action_manager=SimpleNamespace(
            action=torch.tensor([[0.1, 0.0], [0.0, 0.0]]),
            prev_action=torch.zeros(2, 2),
        ),
    )

    torch.testing.assert_close(
        periodic_reward_composition_velocity_cost(env, "twist", axis=0),
        1.0 - torch.exp(-2.0 * torch.tensor([0.05, 0.0])),
    )
    expected_orientation = 1.0 - torch.exp(
        -3.0 * torch.tensor([torch.sin(pitch / 2) ** 2, 0.0])
    )
    torch.testing.assert_close(
        periodic_reward_composition_orientation_cost(env), expected_orientation
    )
    torch.testing.assert_close(
        periodic_reward_composition_action_difference_cost(env),
        1.0 - torch.exp(-5.0 * torch.tensor([0.1, 0.0])),
    )


def test_short_walk_references_use_positive_knee_unloading_at_three_hz() -> None:
    """The fixed-ankle model raises its foot for a positive knee offset."""
    env = SimpleNamespace(
        # 0.280 and 0.300 seconds are the respective mid-swing instants for
        # the 70% and 82% duty references at the 0.33-second gait period.
        episode_length_buf=torch.tensor([280, 300]), step_dt=0.001,
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=torch.zeros(2, 6),
            joint_pos=torch.zeros(2, 6),
        )
    )
    env.scene = {"robot": robot}

    animated = animated_walk_reference_joint_positions(env)
    conservative = conservative_shuffle_reference_joint_positions(env)
    # The exact phase need not be at the sine peak, but both motions must be
    # positive: negative knee motion drives this CAD foot down into the floor.
    assert animated[0, 2] > 0.11
    assert conservative[1, 2] > 0.045
    # Both use the requested maximum 0.33-second per-leg gait cycle.
    assert animated.shape == conservative.shape == (2, 6)


def test_commanded_foot_exchange_scales_with_signed_forward_speed() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            site_pos_w=torch.zeros(1, 2, 3),
            heading_w=torch.zeros(1),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        )
    )
    command = torch.tensor([[0.08, 0.0, 0.0]])
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([1]),
        step_dt=0.1,
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    site_cfg = SimpleNamespace(name="robot", site_ids=[0, 1])

    for speed, difference in ((0.08, 0.012), (0.0, 0.0), (-0.08, -0.012)):
        command[0, 0] = speed
        robot.data.site_pos_w[0, 0, 0] = difference
        torch.testing.assert_close(
            commanded_periodic_foot_exchange(
                env, "twist", period=0.4, asset_cfg=site_cfg
            ),
            torch.ones(1),
        )


def test_commanded_foot_exchange_is_body_heading_relative() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            site_pos_w=torch.zeros(1, 2, 3),
            heading_w=torch.tensor([torch.pi / 2]),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        )
    )
    # At phase 0.25 the desired exchange is +12 mm.  With a 90-degree
    # heading that fore/aft separation lies along world Y, not world X.
    robot.data.site_pos_w[0, 0, 1] = 0.012
    command = torch.tensor([[0.08, 0.0, 0.0]])
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([1]),
        step_dt=0.1,
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    site_cfg = SimpleNamespace(name="robot", site_ids=[0, 1])

    torch.testing.assert_close(
        commanded_periodic_foot_exchange(
            env, "twist", period=0.4, asset_cfg=site_cfg
        ),
        torch.ones(1),
    )


def test_commanded_motion_rate_safety_uses_body_lateral_velocity() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            # World-Y is fast, but body-Y exactly matches the command.
            root_link_lin_vel_w=torch.tensor([[0.0, 1.0, 0.0]]),
            root_link_lin_vel_b=torch.tensor([[0.0, 0.1, 0.0]]),
            root_link_ang_vel_b=torch.zeros(1, 3),
        )
    )
    command = torch.tensor([[0.0, 0.1, 0.0]])
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )

    torch.testing.assert_close(
        commanded_motion_rate_safety_cost(env, "twist"), torch.zeros(1)
    )


def test_lateral_foot_line_term_is_active_only_for_pure_lateral_double_support() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            site_pos_w=torch.tensor([[[0.020, 0.040, 0.0], [0.000, -0.040, 0.0]]]),
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

    error = lateral_foot_line_error(env, "twist", asset_cfg=site_cfg)
    cost = commanded_lateral_foot_line_cost(
        env, "twist", line_scale=0.015, asset_cfg=site_cfg
    )
    torch.testing.assert_close(error, torch.tensor([0.020]))
    assert cost.item() > 0.0

    # Signed metrics isolate the two lateral command directions.
    torch.testing.assert_close(
        lateral_foot_line_error(
            env, "twist", asset_cfg=site_cfg, lateral_direction=1
        ),
        torch.tensor([0.020]),
    )
    torch.testing.assert_close(
        lateral_foot_line_error(
            env, "twist", asset_cfg=site_cfg, lateral_direction=-1
        ),
        torch.zeros(1),
    )

    command[:] = torch.tensor([[0.08, 0.0, 0.0]])
    torch.testing.assert_close(
        lateral_foot_line_error(env, "twist", asset_cfg=site_cfg), torch.zeros(1)
    )
    torch.testing.assert_close(
        commanded_lateral_foot_line_cost(env, "twist", asset_cfg=site_cfg),
        torch.zeros(1),
    )


def test_cycle_displacement_integrates_each_command_step() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.zeros(1, 3), heading_w=torch.zeros(1)
        )
    )
    command = torch.zeros(1, 3)
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.1,
        episode_length_buf=torch.tensor([1]),
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    cost = CommandedCycleDisplacementCost(cfg=None, env=env)
    params = {
        "command_name": "twist",
        "period": 0.25,
        "sagittal_scale": 0.012,
        "lateral_scale": 0.012,
        "yaw_scale": 0.08,
    }

    torch.testing.assert_close(cost(env, **params), torch.zeros(1))
    command[0] = torch.tensor([0.1, 0.0, 0.0])
    robot.data.root_link_pos_w[0, 0] = 0.01
    env.episode_length_buf[:] = 2
    cost(env, **params)
    command[0] = torch.tensor([0.0, 0.1, 0.0])
    robot.data.root_link_pos_w[0, 1] = 0.01
    env.episode_length_buf[:] = 3
    torch.testing.assert_close(cost(env, **params), torch.zeros(1), atol=1e-6, rtol=0)

    command.zero_()
    env.episode_length_buf[:] = 4
    cost(env, **params)
    robot.data.root_link_pos_w[0, 0] += 0.012
    env.episode_length_buf[:] = 5
    torch.testing.assert_close(
        cost(env, **params),
        torch.tensor([2.0**0.5 - 1.0]),
        atol=1e-6,
        rtol=0,
    )


def test_episode_displacement_tracks_expected_world_motion_across_steps() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.zeros(1, 3), heading_w=torch.zeros(1)
        )
    )
    command = torch.zeros(1, 3)
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.1,
        episode_length_buf=torch.tensor([1]),
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    cost = CommandedEpisodeDisplacementCost(cfg=None, env=env)
    params = {
        "command_name": "twist",
        "linear_scale": 0.08,
        "yaw_scale": 0.40,
        "max_cost": 12.0,
        "warmup_s": 0.50,
    }

    # The first call initializes the episode baseline.  Subsequent calls
    # integrate the command over each elapsed step, just as the robot moves.
    torch.testing.assert_close(cost(env, **params), torch.zeros(1))
    command[0] = torch.tensor([0.1, 0.0, 0.0])
    for step in range(2, 12):
        env.episode_length_buf[:] = step
        robot.data.root_link_pos_w[0, 0] = 0.1 * (step - 1) * env.step_dt
        torch.testing.assert_close(cost(env, **params), torch.zeros(1), atol=1e-6, rtol=0)


def test_episode_displacement_catches_persistent_under_travel() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.zeros(1, 3), heading_w=torch.zeros(1)
        )
    )
    command = torch.zeros(1, 3)
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.1,
        episode_length_buf=torch.tensor([1]),
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    cost = CommandedEpisodeDisplacementCost(cfg=None, env=env)
    params = {"command_name": "twist", "warmup_s": 0.50}
    torch.testing.assert_close(cost(env, **params), torch.zeros(1))

    # The commanded displacement accumulates over ten steps while the robot
    # stays still, so this cannot be hidden by a near-zero single-cycle error.
    command[0] = torch.tensor([0.1, 0.0, 0.0])
    env.episode_length_buf[:] = 11
    value = cost(env, **params)
    assert value.item() > 0.0


def test_episode_displacement_resets_its_baseline_on_new_episode() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.zeros(1, 3), heading_w=torch.zeros(1)
        )
    )
    command = torch.tensor([[0.1, 0.0, 0.0]])
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.1,
        episode_length_buf=torch.tensor([1]),
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    cost = CommandedEpisodeDisplacementCost(cfg=None, env=env)
    assert cost(env, "twist").item() == 0.0
    env.episode_length_buf[:] = 2
    robot.data.root_link_pos_w[0, 0] = 0.2
    assert cost(env, "twist").item() > 0.0

    # A reset may reuse the same step counter.  The term must establish a
    # fresh origin and not carry the previous episode's accumulated command.
    cost.reset(torch.tensor([0]))
    env.episode_length_buf[:] = 1
    robot.data.root_link_pos_w[0, 0] = 5.0
    torch.testing.assert_close(cost(env, "twist"), torch.zeros(1))


def test_actuator_effort_joint_weights_reduce_thigh_and_knee_penalties() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            qfrc_actuator=torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        )
    )
    env = SimpleNamespace(scene={"robot": robot})
    baseline = actuator_effort_cost(env, effort_scale=1.5)
    weighted = actuator_effort_cost(
        env,
        effort_scale=1.5,
        joint_weights=(1.0, 0.25, 0.25, 1.0, 0.25, 0.25),
    )
    assert weighted.item() < baseline.item()
    expected = (2.0 + 4.0 * 0.25) / 6.0 * (1.0 - torch.exp(torch.tensor(-1.0 / 1.5)))
    torch.testing.assert_close(weighted, expected[None], atol=1e-6, rtol=0)


def test_bilateral_reference_score_is_limited_by_the_worse_leg() -> None:
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([280]), step_dt=0.001,
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=torch.zeros(1, 6),
            joint_pos=torch.zeros(1, 6),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        )
    )
    env.scene = {"robot": robot}
    target = animated_walk_reference_joint_positions(env)
    robot.data.joint_pos = target.clone()
    both_legs = track_animated_walk_reference(env)
    # Deliberately make the left leg miss its phase target while leaving the
    # right one exact; an all-joint average could partially hide this.
    robot.data.joint_pos[:, 3:] = target[:, 3:] + 1.0
    one_leg = track_animated_walk_reference(env)

    torch.testing.assert_close(both_legs, torch.ones(1))
    assert one_leg.item() < 0.1


def test_unloading_reference_and_force_target_are_low_clearance_but_asymmetric() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=torch.zeros(2, 6),
            joint_pos=torch.zeros(2, 6),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]] * 2),
        )
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            # phase 0: 50/50 double support; phase 1/3: right support at 75/25.
            force=torch.tensor([
                [[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]],
                [[0.0, 0.0, 15.0], [0.0, 0.0, 5.0]],
            ])
        )
    )
    env = SimpleNamespace(
        scene={"robot": robot, "feet_ground_contact": sensor},
        episode_length_buf=torch.tensor([0, 110]),
        step_dt=0.001,
    )
    reference = unloading_walk_reference_joint_positions(env)
    # Compared with the conservative 0.05-rad bump, this task actively uses
    # knee pitch but remains far below the physical knee limit.
    assert torch.max(reference[:, (2, 5)]) > 0.07
    torch.testing.assert_close(
        scheduled_swing_unloading(env, "feet_ground_contact"),
        torch.ones(2), atol=1e-6, rtol=0,
    )


def test_walk_pitch_terms_prioritize_pitch_over_roll_and_yaw() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.2, 0.0, -0.98]]),
            root_link_ang_vel_b=torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]),
        )
    )
    command = torch.zeros(2, 3)
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    assert pitch_upright(env)[0] > pitch_upright(env)[1]
    penalty = pitch_dominant_angular_velocity(env, "twist")
    assert penalty[1] > penalty[0]


def test_periodic_composition_only_costs_force_in_swing_and_speed_in_stance() -> None:
    # At phase 0.30, right is in stance and left is in swing for a 70% duty gait.
    robot = SimpleNamespace(
        data=SimpleNamespace(
            site_lin_vel_w=torch.tensor([[[0.20, 0.0, 0.0], [0.00, 0.0, 0.0]]]),
        )
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(force=torch.tensor([[[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]]]))
    )
    env = SimpleNamespace(
        scene={"robot": robot, "feet_ground_contact": sensor},
        episode_length_buf=torch.tensor([99]), step_dt=0.001,
    )
    swing = smooth_swing_probability(env)
    assert swing[0, 0] < 0.01
    assert swing[0, 1] > 0.90
    # Right force is allowed in stance, left is unloaded in swing. Conversely,
    # the right stance-foot motion is expensive while left swing motion is free.
    assert periodic_swing_force_cost(env, "feet_ground_contact").item() < 0.01
    assert periodic_stance_foot_speed_cost(env).item() > 0.90


def test_common_skills_enable_full_sim2real_randomization_by_default() -> None:
    assert sim2real_randomization_scale(SimpleNamespace(cfg=SimpleNamespace())) == 1.0
    assert sim2real_randomization_scale(
        SimpleNamespace(cfg=SimpleNamespace(sim2real_randomization_enabled=False))
    ) == 0.0


def test_direction_progress_does_not_reward_a_stationary_robot() -> None:
    command = torch.tensor([[0.2, 0.0, 0.0]] * 3)
    actual_velocity = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(root_link_lin_vel_b=actual_velocity)
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )

    reward = command_direction_progress(env, "twist")

    torch.testing.assert_close(reward, torch.tensor([0.0, 0.5, -0.5]))


def test_angular_velocity_penalty_allows_only_commanded_yaw() -> None:
    command = torch.tensor(
        [[0.2, 0.0, 0.4], [0.2, 0.0, 0.4], [0.2, 0.0, 0.4]]
    )
    angular_velocity = torch.tensor(
        [[0.0, 0.0, 0.4], [1.0, -0.5, 0.4], [0.0, 0.0, -0.4]]
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(root_link_ang_vel_b=angular_velocity)
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )

    penalty = uncommanded_base_angular_velocity(env, "twist")

    torch.testing.assert_close(penalty, torch.tensor([0.0, 1.25, 0.64]))


def test_straight_path_penalties_measure_heading_and_lateral_drift() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            heading_w=torch.tensor([0.0, torch.pi / 2]),
            root_link_pos_w=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.15, 0.0]]),
        )
    )
    class Scene(dict):
        pass

    scene = Scene(robot=robot)
    scene.env_origins = torch.zeros(2, 3)
    env = SimpleNamespace(scene=scene)

    torch.testing.assert_close(
        fixed_heading_error(env), torch.tensor([0.0, 1.0])
    )
    torch.testing.assert_close(lateral_path_deviation(env), torch.tensor([0.0, 1.0]))


def test_unwrapped_heading_penalty_accumulates_across_pi() -> None:
    robot = SimpleNamespace(data=SimpleNamespace(heading_w=torch.tensor([3.10])))
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        episode_length_buf=torch.tensor([1]),
        scene={"robot": robot},
    )
    penalty = UnwrappedFixedHeadingError(cfg=None, env=env)

    torch.testing.assert_close(
        penalty(env, max_penalty=100.0), torch.tensor([3.10**2]), atol=1e-5, rtol=0
    )

    env.episode_length_buf[:] = 2
    robot.data.heading_w[:] = -3.10
    torch.testing.assert_close(
        penalty(env, max_penalty=100.0),
        torch.tensor([(3.10 + 0.0831853) ** 2]),
        atol=1e-5,
        rtol=0,
    )


def test_coupled_yaw_cost_uses_relative_deadband_without_straight_yaw_penalty() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            heading_w=torch.zeros(2),
            root_link_lin_vel_w=torch.tensor([[0.1, 0.0], [0.1, 0.0]]),
            root_link_ang_vel_b=torch.zeros(2, 3),
        )
    )
    command = torch.tensor([[0.1, 0.0, 0.03], [0.1, 0.0, 0.0]])
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    cost = instantaneous_world_command_velocity_cost(
        env,
        "twist",
        yaw_deadband=0.02,
        yaw_relative_scale=0.5,
        yaw_scale_floor=0.05,
        coupled_vx_threshold=0.02,
    )
    assert cost[0] > 0.1
    torch.testing.assert_close(cost[1], torch.tensor(0.0))


def test_gradual_heading_drift_cost_uses_initial_heading_deadband_and_ramp() -> None:
    robot = SimpleNamespace(data=SimpleNamespace(heading_w=torch.tensor([1.20])))
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        common_step_counter=0,
        episode_length_buf=torch.tensor([1]),
        scene={"robot": robot},
    )
    cost = GradualBoundedHeadingDriftCost(cfg=None, env=env)
    params = {
        "deadband": 0.05,
        "scale": 0.20,
        "max_cost": 4.0,
        "ramp_steps": 100,
    }

    torch.testing.assert_close(cost(env, **params), torch.zeros(1))
    env.common_step_counter = 50
    env.episode_length_buf[:] = 2
    robot.data.heading_w[:] = 1.25
    torch.testing.assert_close(cost(env, **params), torch.zeros(1), atol=1e-6, rtol=0)
    env.episode_length_buf[:] = 3
    robot.data.heading_w[:] = 1.35
    expected = 0.5 * 4.0 * (1.0 - torch.exp(torch.tensor(-0.25 / 4.0)))
    torch.testing.assert_close(cost(env, **params), expected[None], atol=1e-6, rtol=0)


def test_cycle_heading_gate_keeps_accumulating_after_a_full_turn() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
            heading_w=torch.tensor([0.0]),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor([1]),
        scene={"robot": robot},
    )
    gate = CycleNetMotionGate(cfg=None, env=env)
    params = {
        "period": 0.33,
        "heading_std": 10.0,
        "cycle_lateral_velocity_std": 10.0,
        "cycle_yaw_rate_std": 10.0,
    }

    torch.testing.assert_close(gate(env, **params), torch.ones(1))
    env.episode_length_buf[:] = 2
    robot.data.heading_w[:] = 3.10
    gate(env, **params)
    env.episode_length_buf[:] = 3
    robot.data.heading_w[:] = -3.10
    gate(env, **params)
    env.episode_length_buf[:] = 4
    robot.data.heading_w[:] = 0.0
    torch.testing.assert_close(
        gate(env, **params), torch.exp(-torch.square(torch.tensor([2.0 * torch.pi / 10.0]))),
        atol=1e-5,
        rtol=0,
    )


def test_cycle_motion_terms_ignore_within_cycle_sway_and_measure_net_drift() -> None:
    """Only a completed 0.33 s cycle contributes to the drift cost."""
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
            heading_w=torch.tensor([0.0]),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor([1]),
        scene={"robot": robot},
    )
    lateral_cost = CycleNetLateralVelocityCost(cfg=None, env=env)
    yaw_cost = CycleNetYawRateCost(cfg=None, env=env)

    # Start of the first cycle: no completed-cycle penalty exists yet.
    torch.testing.assert_close(
        lateral_cost(env, period=0.33, std=0.02), torch.zeros(1)
    )
    torch.testing.assert_close(yaw_cost(env, period=0.33, std=0.25), torch.zeros(1))

    # At the next cycle boundary, 6.6 mm net lateral drift and 0.0825 rad net
    # yaw have accumulated; the terms report their corresponding mean rates.
    env.episode_length_buf[:] = 17
    robot.data.root_link_pos_w[:, 1] = 0.0066
    robot.data.heading_w[:] = 0.0825
    torch.testing.assert_close(
        lateral_cost(env, period=0.33, std=0.02), torch.tensor([1.0]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        yaw_cost(env, period=0.33, std=0.25), torch.tensor([1.0]), atol=1e-6, rtol=0
    )

    # An asynchronous reset clears its history instead of carrying old drift
    # into the fresh episode.
    lateral_cost.reset(torch.tensor([0]))
    yaw_cost.reset(torch.tensor([0]))
    env.episode_length_buf[:] = 1
    robot.data.root_link_pos_w[:, 1] = 0.5
    robot.data.heading_w[:] = 0.0
    torch.testing.assert_close(
        lateral_cost(env, period=0.33, std=0.02), torch.zeros(1)
    )


def test_commanded_cycle_motion_tracks_piecewise_lateral_commands() -> None:
    command = torch.tensor([[0.05, 0.04, 0.0]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.zeros(1, 3),
            heading_w=torch.zeros(1),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor([1]),
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    cost = CommandedCyclePlanarVelocityCost(cfg=None, env=env)
    params = {"command_name": "twist", "period": 0.33}
    torch.testing.assert_close(cost(env, **params), torch.zeros(1))
    for step in range(2, 18):
        env.episode_length_buf[:] = step
        robot.data.root_link_pos_w[:, :2] = command[:, :2] * ((step - 1) * env.step_dt)
        result = cost(env, **params)
    torch.testing.assert_close(result, torch.zeros(1), atol=1e-6, rtol=0)

    cost.reset(torch.tensor([0]))
    command[:, 1] = 0.0
    env.episode_length_buf[:] = 1
    robot.data.root_link_pos_w[:] = 0.0
    cost(env, **params)
    for step in range(2, 18):
        env.episode_length_buf[:] = step
        elapsed = (step - 1) * env.step_dt
        robot.data.root_link_pos_w[:, 0] = command[:, 0] * elapsed
        robot.data.root_link_pos_w[:, 1] = 0.0066 * elapsed / 0.32
        result = cost(env, **params)
    torch.testing.assert_close(result, torch.ones(1), atol=1e-5, rtol=0)


def test_commanded_path_cost_reduces_to_existing_y_error_at_zero_vy() -> None:
    class Scene(dict):
        pass

    command = torch.tensor([[0.05, 0.0, 0.0]])
    robot = SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.001, 0.08, 0.0]]))
    )
    scene = Scene(robot=robot)
    scene.env_origins = torch.zeros(1, 3)
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor([1]),
        scene=scene,
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    cost = CommandedPlanarPathCost(cfg=None, env=env)
    torch.testing.assert_close(
        cost(env, command_name="twist", lateral_tolerance=0.08),
        torch.ones(1),
    )
