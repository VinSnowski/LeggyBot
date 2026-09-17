"""Exercise real simulator fields; mocks cannot catch selector/axis errors."""
from types import SimpleNamespace

import torch
import pytest
from locomotion.task import make_env_cfg
from locomotion.tasks.gait import (
    symmetric_domain_randomization, full_domain_randomization,
    _interpolated_full_domain_randomization, STS3215_RANDOMIZATION,
    FOOT_GROUND_RANDOMIZATION, _global_geom_ids,
    sim2real_joint_pos,
    Sim2RealJointVelocity,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp.dr.body import _reconstruct_pseudo_inertia_J

@pytest.fixture(scope='module')
def env():
    from mjlab.envs import ManagerBasedRlEnv
    cfg = make_env_cfg('NaturalOmnidirectionalWalkingTaskV3', num_envs=4, seed=123)
    result = ManagerBasedRlEnv(cfg, device='cpu')
    result.reset()
    yield result
    result.close()

FIELDS = ('body_mass', 'body_ipos', 'body_inertia', 'body_iquat')

def snapshot(env, ids):
    bodies = env.scene['robot'].indexing.body_ids
    return [getattr(env.sim.model, f)[ids[:, None], bodies].clone() for f in FIELDS]

def physical(values):
    return _reconstruct_pseudo_inertia_J(*(v.double() for v in values))


def foot_and_terrain_geom_ids(env):
    feet = _global_geom_ids(
        env,
        SceneEntityCfg(
            "robot",
            geom_names=("right_foot_collision", "left_foot_collision"),
            preserve_order=True,
        ),
    )
    terrain = _global_geom_ids(
        env,
        SceneEntityCfg("terrain", geom_names=("terrain",), preserve_order=True),
    )
    return feet, terrain

def test_paired_restores_legs_and_preserves_torso_lateral_com(env):
    ids = torch.arange(4)
    bodies = env.scene['robot'].indexing.body_ids
    full_domain_randomization(env, ids)
    symmetric_domain_randomization(env, ids)
    for field in FIELDS:
        torch.testing.assert_close(
            getattr(env.sim.model, field)[ids[:, None], bodies[1:]],
            env.sim.get_default_field(field)[bodies[1:]].expand(4, *env.sim.get_default_field(field)[bodies[1:]].shape),
        )
    torch.testing.assert_close(env.sim.model.body_ipos[ids, bodies[0], 1],
                              env.sim.get_default_field('body_ipos')[bodies[0], 1].expand(4))
    for name in ('actuator_gainprm','actuator_biasprm','actuator_forcerange'):
        values = getattr(env.sim.model, name)
        torch.testing.assert_close(values[ids[:, None], torch.arange(3)], values[ids[:, None], torch.arange(3,6)])
    dofs = env.scene['robot'].indexing.joint_v_adr
    for name in ('dof_damping', 'dof_frictionloss', 'dof_armature'):
        values = getattr(env.sim.model, name)
        torch.testing.assert_close(values[ids[:, None], dofs[:3]], values[ids[:, None], dofs[3:]])
    bias = env.scene['robot'].data.encoder_bias
    torch.testing.assert_close(bias[:, :3] * torch.tensor([-1.,1.,1.]), bias[:, 3:])
    scale = env.scene['robot'].data.encoder_scale
    torch.testing.assert_close(scale[:, :3], scale[:, 3:])
    action = env.action_manager.get_term('joint_pos')
    torch.testing.assert_close(
        action._firmware_max_velocity[:, :3],
        action._firmware_max_velocity[:, 3:],
    )
    torch.testing.assert_close(
        action._firmware_max_acceleration[:, :3],
        action._firmware_max_acceleration[:, 3:],
    )
    backlash = env.action_manager.get_term('joint_pos')._backlash_half_width
    torch.testing.assert_close(backlash[:, :3], backlash[:, 3:])


def test_paired_foot_geometry_is_mirrored_and_contact_material_is_shared(env):
    ids = torch.arange(4)
    env.cfg.domain_randomization_scale = 1.0
    torch.manual_seed(84)
    symmetric_domain_randomization(env, ids)
    feet, _ = foot_and_terrain_geom_ids(env)
    env_grid, foot_grid = torch.meshgrid(ids, feet, indexing="ij")
    model = env.sim.model
    defaults = env.sim.get_default_field

    position_delta = (
        model.geom_pos[env_grid, foot_grid]
        - defaults("geom_pos")[feet].unsqueeze(0)
    )
    torch.testing.assert_close(
        position_delta[..., :2], torch.zeros_like(position_delta[..., :2])
    )
    torch.testing.assert_close(position_delta[:, 0, 2], position_delta[:, 1, 2])

    # Nominal foot collision quaternions are identity. Equal local-X and
    # opposite local-Y rotations produce [w, x, -y, -z] under reflection.
    quat = model.geom_quat[env_grid, foot_grid]
    torch.testing.assert_close(quat[:, 0, :2], quat[:, 1, :2])
    torch.testing.assert_close(quat[:, 0, 2:], -quat[:, 1, 2:])
    torch.testing.assert_close(
        torch.linalg.vector_norm(quat, dim=-1), torch.ones((4, 2))
    )

    torch.testing.assert_close(
        model.geom_friction[env_grid[:, 0], foot_grid[:, 0]],
        model.geom_friction[env_grid[:, 1], foot_grid[:, 1]],
    )
    torch.testing.assert_close(
        model.geom_solref[env_grid[:, 0], foot_grid[:, 0]],
        model.geom_solref[env_grid[:, 1], foot_grid[:, 1]],
    )
    torch.testing.assert_close(
        model.geom_solimp[env_grid[:, 0], foot_grid[:, 0]],
        model.geom_solimp[env_grid[:, 1], foot_grid[:, 1]],
    )


def test_full_foot_ground_prior_ranges_and_collision_only_isolation(env):
    ids = torch.arange(4)
    env.cfg.domain_randomization_scale = 1.0
    feet, terrain = foot_and_terrain_geom_ids(env)
    robot_geoms = env.scene["robot"].indexing.geom_ids
    unchanged = robot_geoms[~torch.isin(robot_geoms, feet)]
    defaults = env.sim.get_default_field
    torch.manual_seed(251)
    full_domain_randomization(env, ids, randomize_latency=False)
    model = env.sim.model
    env_grid, foot_grid = torch.meshgrid(ids, feet, indexing="ij")

    position_delta = (
        model.geom_pos[env_grid, foot_grid]
        - defaults("geom_pos")[feet].unsqueeze(0)
    )
    prior = FOOT_GROUND_RANDOMIZATION
    assert (
        float(position_delta[..., 2].abs().max())
        <= prior.sole_height_m + 1e-8
    )
    assert torch.any(
        torch.abs(position_delta[:, 0, 2] - position_delta[:, 1, 2]) > 1e-8
    )

    solref = model.geom_solref[env_grid, foot_grid]
    assert float(solref[..., 0].min()) >= prior.contact_time_constant_s[0]
    assert float(solref[..., 0].max()) <= prior.contact_time_constant_s[1]
    assert float(solref[..., 1].min()) >= prior.contact_damping_ratio[0]
    assert float(solref[..., 1].max()) <= prior.contact_damping_ratio[1]
    width = model.geom_solimp[env_grid, foot_grid, 2]
    assert float(width.min()) >= prior.compliance_width_m[0]
    assert float(width.max()) <= prior.compliance_width_m[1]

    friction = model.geom_friction[env_grid, foot_grid]
    assert float(friction[..., 0].min()) >= 0.5
    assert float(friction[..., 0].max()) <= 1.2
    default_friction = defaults("geom_friction")[feet]
    torsion_factor = friction[..., 1] / default_friction[None, :, 1]
    rolling_factor = friction[..., 2] / default_friction[None, :, 2]
    assert float(torsion_factor.min()) >= prior.torsional_friction_factor[0]
    assert float(torsion_factor.max()) <= prior.torsional_friction_factor[1]
    assert float(rolling_factor.min()) >= prior.rolling_friction_factor[0]
    assert float(rolling_factor.max()) <= prior.rolling_friction_factor[1]

    # Only the two collision soles and the terrain orientation may change.
    for field in ("geom_pos", "geom_quat"):
        torch.testing.assert_close(
            getattr(model, field)[ids[:, None], unchanged],
            defaults(field)[unchanged].unsqueeze(0).expand(4, -1, -1),
        )
    floor_quat = model.geom_quat[ids[:, None], terrain]
    torch.testing.assert_close(
        torch.linalg.vector_norm(floor_quat, dim=-1), torch.ones((4, 1))
    )


def test_zero_randomization_strength_restores_nominal_foot_and_ground(env):
    ids = torch.arange(4)
    feet, terrain = foot_and_terrain_geom_ids(env)
    selected_geoms = torch.cat((feet, terrain))
    env.cfg.domain_randomization_scale = 0.0
    try:
        full_domain_randomization(env, ids, randomize_latency=False)
        for field in (
            "geom_pos", "geom_quat", "geom_friction", "geom_solref", "geom_solimp"
        ):
            torch.testing.assert_close(
                getattr(env.sim.model, field)[ids[:, None], selected_geoms],
                env.sim.get_default_field(field)[selected_geoms]
                .unsqueeze(0)
                .expand(4, -1, -1),
            )
    finally:
        env.cfg.domain_randomization_scale = 1.0

def test_ramp_interpolates_physics_and_is_history_independent(env):
    ids = torch.tensor([0,2])
    ramped_model_fields = (
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "dof_damping",
        "dof_frictionloss",
        "dof_armature",
    )
    feet, terrain = foot_and_terrain_geom_ids(env)
    env_grid, foot_grid = torch.meshgrid(ids, feet, indexing="ij")
    model = env.sim.model
    torch.manual_seed(9)
    symmetric_domain_randomization(env, ids)
    paired = physical(snapshot(env, ids))
    paired_model = {
        name: getattr(model, name)[ids].clone() for name in ramped_model_fields
    }
    paired_foot_pos = model.geom_pos[env_grid, foot_grid].clone()
    paired_foot_quat = model.geom_quat[env_grid, foot_grid].clone()
    paired_bias = env.scene['robot'].data.encoder_bias[ids].clone()
    paired_scale = env.scene['robot'].data.encoder_scale[ids].clone()
    action = env.action_manager.get_term('joint_pos')
    paired_backlash = action._backlash_half_width[ids].clone()
    paired_profile_velocity = action._firmware_max_velocity[ids].clone()
    paired_profile_acceleration = action._firmware_max_acceleration[ids].clone()
    paired_latency = action._base_delay_steps[ids].clone()
    full_domain_randomization(env, ids, randomize_latency=False)
    full = physical(snapshot(env, ids))
    full_model = {
        name: getattr(model, name)[ids].clone() for name in ramped_model_fields
    }
    full_foot_pos = model.geom_pos[env_grid, foot_grid].clone()
    full_foot_quat = model.geom_quat[env_grid, foot_grid].clone()
    # These effects are already full-strength during the paired phase and are
    # deliberately resampled, rather than amplitude-ramped, at every reset.
    full_contact = {
        name: getattr(model, name)[env_grid, foot_grid].clone()
        for name in ("geom_friction", "geom_solref", "geom_solimp")
    }
    full_floor_quat = model.geom_quat[ids[:, None], terrain].clone()
    full_bias = env.scene['robot'].data.encoder_bias[ids].clone()
    full_scale = env.scene['robot'].data.encoder_scale[ids].clone()
    full_backlash = action._backlash_half_width[ids].clone()
    full_profile_velocity = action._firmware_max_velocity[ids].clone()
    full_profile_acceleration = action._firmware_max_acceleration[ids].clone()
    before_other = snapshot(env, torch.tensor([1,3]))
    torch.manual_seed(9)
    _interpolated_full_domain_randomization(env, ids, .5)
    actual = physical(snapshot(env, ids))
    torch.testing.assert_close(actual, (paired+full)/2, atol=1e-8, rtol=2e-5)
    for name in ramped_model_fields:
        torch.testing.assert_close(
            getattr(model, name)[ids],
            (paired_model[name] + full_model[name]) / 2,
        )
    torch.testing.assert_close(
        model.geom_pos[env_grid, foot_grid],
        (paired_foot_pos + full_foot_pos) / 2,
    )
    same_hemisphere = (paired_foot_quat * full_foot_quat).sum(dim=-1) >= 0
    aligned_full_quat = torch.where(
        same_hemisphere[..., None], full_foot_quat, -full_foot_quat
    )
    expected_foot_quat = torch.nn.functional.normalize(
        (paired_foot_quat + aligned_full_quat) / 2, dim=-1
    )
    torch.testing.assert_close(
        model.geom_quat[env_grid, foot_grid], expected_foot_quat
    )
    for name, expected in full_contact.items():
        torch.testing.assert_close(
            getattr(model, name)[env_grid, foot_grid], expected
        )
    torch.testing.assert_close(
        model.geom_quat[ids[:, None], terrain], full_floor_quat
    )
    torch.testing.assert_close(
        env.scene['robot'].data.encoder_bias[ids], (paired_bias + full_bias) / 2
    )
    torch.testing.assert_close(
        env.scene['robot'].data.encoder_scale[ids], (paired_scale + full_scale) / 2
    )
    torch.testing.assert_close(
        action._backlash_half_width[ids],
        (paired_backlash + full_backlash) / 2,
    )
    torch.testing.assert_close(
        action._firmware_max_velocity[ids],
        (paired_profile_velocity + full_profile_velocity) / 2,
    )
    torch.testing.assert_close(
        action._firmware_max_acceleration[ids],
        (paired_profile_acceleration + full_profile_acceleration) / 2,
    )
    torch.testing.assert_close(action._base_delay_steps[ids], paired_latency)
    for a,b in zip(before_other,snapshot(env,torch.tensor([1,3]))):
        torch.testing.assert_close(a,b)
    full_domain_randomization(env, ids)
    torch.manual_seed(9)
    _interpolated_full_domain_randomization(env, ids, .5)
    torch.testing.assert_close(physical(snapshot(env,ids)), actual, atol=1e-8,rtol=2e-5)

@pytest.mark.parametrize('scale,maximum', [(0.,0),(.25,1),(1.,3)])
def test_ramp_latency_respects_strength(env, monkeypatch, scale, maximum):
    monkeypatch.setattr(env.cfg, 'domain_randomization_scale', scale, raising=False)
    samples=[]
    action = env.action_manager.get_term('joint_pos')
    monkeypatch.setattr(
        action,
        'set_command_delay',
        lambda lags, strength, ids: samples.append((lags.clone(), strength)),
    )
    _interpolated_full_domain_randomization(env, torch.arange(4), .999)
    assert len(samples) == 1
    assert int(samples[0][0].max()) <= maximum
    assert samples[0][1] == scale


def test_full_servo_prior_stays_in_declared_envelope_and_is_independent(env):
    ids = torch.arange(4)
    env.cfg.domain_randomization_scale = 1.0
    torch.manual_seed(1234)
    full_domain_randomization(env, ids, randomize_latency=False)
    defaults = env.sim.get_default_field('actuator_forcerange')[:, 1]
    strength = env.sim.model.actuator_forcerange[ids, :, 1] / defaults
    lower = STS3215_RANDOMIZATION.supply_factor[0] * STS3215_RANDOMIZATION.unit_torque_factor[0]
    upper = STS3215_RANDOMIZATION.supply_factor[1] * STS3215_RANDOMIZATION.unit_torque_factor[1]
    assert torch.all((strength >= lower) & (strength <= upper))
    assert torch.any(torch.abs(strength[:, :3] - strength[:, 3:]) > 1e-5)

    calibration_scale = env.scene['robot'].data.encoder_scale[ids]
    assert float(calibration_scale.min()) >= STS3215_RANDOMIZATION.calibration_scale_factor[0]
    assert float(calibration_scale.max()) <= STS3215_RANDOMIZATION.calibration_scale_factor[1]
    assert torch.any(
        torch.abs(calibration_scale[:, :3] - calibration_scale[:, 3:]) > 1e-6
    )
    action = env.action_manager.get_term('joint_pos')
    assert torch.any(
        torch.abs(
            action._firmware_max_acceleration[ids, :3]
            - action._firmware_max_acceleration[ids, 3:]
        )
        > 1e-5
    )

    backlash = env.action_manager.get_term('joint_pos')._backlash_half_width[ids]
    assert float(backlash.min()) >= STS3215_RANDOMIZATION.backlash_total_rad[0] / 2
    assert float(backlash.max()) <= STS3215_RANDOMIZATION.backlash_total_rad[1] / 2
    assert torch.any(torch.abs(backlash[:, :3] - backlash[:, 3:]) > 1e-5)


def test_policy_targets_use_sts3215_position_counts(env):
    action = env.action_manager.get_term("joint_pos")
    action.set_backlash_half_width(torch.zeros((4, 6)), torch.arange(4))
    action.process_actions(
        torch.tensor([[.123, -.456, .789, -.234, .567, -.891]]).repeat(4, 1)
    )
    step = STS3215_RANDOMIZATION.position_resolution_rad
    target = action._processed_actions
    torch.testing.assert_close(
        target / step,
        torch.round(target / step),
    )


def test_calibration_is_shared_by_command_and_feedback(env):
    action = env.action_manager.get_term("joint_pos")
    ids = torch.arange(4)
    bias = torch.tensor([[.01, -.02, .03, -.04, .05, -.06]]).repeat(4, 1)
    scale = torch.tensor([[1.002, .998, 1.004, .997, 1.001, .996]]).repeat(4, 1)
    action.set_servo_calibration(bias, scale, ids)
    software_target = torch.tensor(
        [[.10, -.20, .30, -.15, .25, -.35]], device=env.device
    ).repeat(4, 1)
    physical_target = action._software_to_physical(software_target)
    recovered = physical_target * scale + bias
    torch.testing.assert_close(recovered, software_target)


def test_command_fifo_preserves_packet_order_and_synchrony(env, monkeypatch):
    action = env.action_manager.get_term("joint_pos")
    ids = torch.arange(4)
    original_counter = env._sim_step_counter
    try:
        action.reset(ids)
        action.set_servo_calibration(torch.zeros((4, 6)), torch.ones((4, 6)), ids)
        action.set_backlash_half_width(torch.zeros((4, 6)), ids)
        action.set_command_delay(torch.tensor([7, 7, 7, 7]), 1.0, ids)
        sampled_delay = torch.full((4,), 7)
        monkeypatch.setattr(action, "_sample_command_delay", lambda: sampled_delay)

        action.process_actions(torch.full((4, 6), 0.25))
        first = action._delay_targets[:, 0].clone()
        first_deadline = action._delay_deadlines[:, 0].clone()
        env._sim_step_counter = original_counter + 4
        sampled_delay.fill_(0)
        action.process_actions(torch.full((4, 6), -0.25))
        second = action._delay_targets[:, 1].clone()
        # The nominally earlier second arrival is clamped behind the first
        # serial packet instead of overtaking it.
        torch.testing.assert_close(action._delay_deadlines[:, 1], first_deadline)
        assert torch.all(first != second)

        env._sim_step_counter = int(first_deadline[0]) - 1
        action.apply_actions()
        assert torch.all(action._delay_count == 2)
        env._sim_step_counter += 1
        action.apply_actions()
        assert torch.all(action._delay_count == 0)
        torch.testing.assert_close(action._delivered_target, second)

        sampled_delay.fill_(3)
        action.process_actions(torch.full((4, 6), 0.1))
        action.reset(torch.tensor([0, 2]))
        torch.testing.assert_close(
            action._delay_count, torch.tensor([0, 1, 0, 1])
        )
    finally:
        env._sim_step_counter = original_counter
        action.reset(ids)


def test_training_firmware_profile_limits_delivered_target_step(env):
    action = env.action_manager.get_term("joint_pos")
    ids = torch.arange(4)
    action.reset(ids)
    start = action._firmware_position.clone()
    action._delivered_target[:] = start + 0.1
    action.set_firmware_profile_limits(
        torch.ones((4, 6), device=env.device),
        torch.full((4, 6), 2.0, device=env.device),
        ids,
    )

    action._advance_firmware_profile()

    dt = env.physics_dt
    torch.testing.assert_close(
        action._firmware_velocity,
        torch.full_like(action._firmware_velocity, 2.0 * dt),
    )
    torch.testing.assert_close(
        action._firmware_position,
        start + torch.full_like(start, 2.0 * dt * dt),
    )
    assert torch.all(action._firmware_position < action._delivered_target)
    action.reset(ids)


def test_actor_joint_feedback_uses_sts3215_position_counts(env, monkeypatch):
    monkeypatch.setattr(env.cfg, "domain_randomization_scale", 0.0, raising=False)
    physical_position = env.scene["robot"].data.joint_pos.clone()
    observation = sim2real_joint_pos(env)
    default = env.scene["robot"].data.default_joint_pos
    step = STS3215_RANDOMIZATION.position_resolution_rad
    observed_absolute = observation + default
    torch.testing.assert_close(
        observed_absolute / step,
        torch.round(observed_absolute / step),
    )
    torch.testing.assert_close(env.scene["robot"].data.joint_pos, physical_position)


def test_training_velocity_is_derived_from_policy_rate_position_samples():
    position = torch.zeros((2, 6))
    fake_env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        cfg=SimpleNamespace(domain_randomization_scale=0.0),
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    joint_pos=position,
                    joint_pos_biased=position,
                )
            )
        },
    )
    term = Sim2RealJointVelocity(
        SimpleNamespace(params={"filter_alpha": 0.5}), fake_env
    )
    torch.testing.assert_close(term(fake_env), torch.zeros_like(position))

    step = STS3215_RANDOMIZATION.position_resolution_rad
    position.add_(2.0 * step)
    fake_env.episode_length_buf.add_(1)
    expected = torch.full_like(position, 0.5 * 2.0 * step / fake_env.step_dt)
    torch.testing.assert_close(term(fake_env), expected)

    # A repeated observation-manager query in the same environment step must
    # not differentiate the position a second time.
    position.add_(10.0 * step)
    torch.testing.assert_close(term(fake_env), expected)

    term.reset(torch.tensor([0]))
    fake_env.episode_length_buf[0] = 0
    result = term(fake_env)
    torch.testing.assert_close(result[0], torch.zeros(6))
