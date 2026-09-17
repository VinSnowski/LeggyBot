from pathlib import Path

import mujoco
import numpy as np
import pytest

from locomotion.play_native import _joint_addresses, reset_robot
from locomotion.robot_profiles import (
    LEGGYBOT_FIXED_ANKLE_PROFILE,
    LEGGYBOT_PROFILE,
    make_native_model,
)


MODEL_PATH = (
    Path(__file__).parents[1]
    / "locomotion"
    / "LeggyBot"
    / "mjcf"
    / "LeggyBot.xml"
)
NEW_MODEL_PATH = (
    Path(__file__).parents[1]
    / "locomotion"
    / "NewLeggyBot"
    / "mjcf"
    / "NewLeggyBot.xml"
)

# The legacy CAD model is optional in the standalone LeggyBot repository;
# current training uses NewLeggyBot exclusively.
AVAILABLE_MODEL_PATHS = tuple(
    path for path in (MODEL_PATH, NEW_MODEL_PATH) if path.is_file()
)

JOINTS = (
    "RightBaseToHip",
    "RightHipToThigh",
    "RightKnee",
    "LeftBaseToHip",
    "LeftHipToThigh",
    "LeftKnee",
)


def _joint_dof(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id >= 0
    return int(model.jnt_dofadr[joint_id])


@pytest.mark.parametrize("model_path", AVAILABLE_MODEL_PATHS)
def test_sts3215_baseline_is_applied_to_every_joint(model_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))

    assert model.nu == len(JOINTS)
    for name in JOINTS:
        dof = _joint_dof(model, name)
        assert model.dof_damping[dof] == 0.60
        assert model.dof_frictionloss[dof] == 0.052
        assert model.dof_armature[dof] == 0.028

    np.testing.assert_allclose(model.actuator_gainprm[:, 0], 17.8)
    np.testing.assert_allclose(model.actuator_biasprm[:, 1], -17.8)
    np.testing.assert_allclose(model.actuator_biasprm[:, 2], 0.0)
    np.testing.assert_allclose(
        model.actuator_forcerange,
        np.tile(np.array([-2.942, 2.942]), (len(JOINTS), 1)),
    )


@pytest.mark.parametrize("model_path", AVAILABLE_MODEL_PATHS)
def test_sts3215_force_saturates_and_dynamics_remain_finite(
    model_path: Path,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    weld_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, "inspection_base_weld"
    )
    data.eq_active[weld_id] = True

    # A deliberately large target error must saturate at nominal stall torque.
    data.ctrl[:] = model.actuator_ctrlrange[:, 1]
    observed_peak = 0.0
    for _ in range(round(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        observed_peak = max(observed_peak, float(np.abs(data.actuator_force).max()))

    assert observed_peak <= 2.942 + 1e-9
    assert observed_peak >= 2.942 - 1e-6
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()


def test_root_frame_uses_forward_x_and_left_y() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH if MODEL_PATH.is_file() else NEW_MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def site_pos(name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        return data.site_xpos[site_id]

    right = site_pos("right_foot")
    left = site_pos("left_foot")
    imu = site_pos("imu")

    # Feet are side-by-side on Y and their toe direction is local +X.
    assert right[1] < imu[1] < left[1]
    assert right[0] > imu[0]
    assert left[0] > imu[0]
    np.testing.assert_allclose(model.body_ipos[1], (-0.0239, -0.0001815, 0.48454))


@pytest.mark.parametrize("model_path", AVAILABLE_MODEL_PATHS)
def test_floating_base_origin_is_on_the_sagittal_plane(model_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "BaseLink")
    right_hip_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "RightHip"
    )
    left_hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LeftHip")
    right_foot_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "right_foot"
    )
    left_foot_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "left_foot"
    )

    root_y = data.xpos[root_id, 1]
    assert 0.5 * (
        data.xpos[right_hip_id, 1] + data.xpos[left_hip_id, 1]
    ) == pytest.approx(root_y, abs=1e-12)
    if model_path == NEW_MODEL_PATH:
        # The current split-shin/foot model also has exactly mirrored foot
        # kinematics.  The legacy combined shin/foot CAD retains its historical
        # 2.475 mm foot-center offset even though its base frame is centered.
        assert 0.5 * (
            data.site_xpos[right_foot_id, 1] + data.site_xpos[left_foot_id, 1]
        ) == pytest.approx(root_y, abs=1e-12)


def _whole_body_com_offset_from_feet(
    model: mujoco.MjModel, data: mujoco.MjData
) -> float:
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "BaseLink")
    foot_x = []
    for name in ("right_foot", "left_foot"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        foot_x.append(data.site_xpos[site_id, 0])
    return float(data.subtree_com[root_id, 0] - np.mean(foot_x))


def test_fixed_ankle_model_uses_the_new_centered_foot() -> None:
    pitched = make_native_model(LEGGYBOT_FIXED_ANKLE_PROFILE)
    pitched_data = mujoco.MjData(pitched)
    pitched_qpos, _ = _joint_addresses(pitched, LEGGYBOT_FIXED_ANKLE_PROFILE)
    reset_robot(pitched, pitched_data, pitched_qpos, LEGGYBOT_FIXED_ANKLE_PROFILE)

    pitched_offset = _whole_body_com_offset_from_feet(pitched, pitched_data)
    assert abs(pitched_offset) < 0.005

    if MODEL_PATH.is_file():
        original = make_native_model(LEGGYBOT_PROFILE)
        original_data = mujoco.MjData(original)
        original_qpos, _ = _joint_addresses(original, LEGGYBOT_PROFILE)
        reset_robot(original, original_data, original_qpos, LEGGYBOT_PROFILE)
        original_offset = _whole_body_com_offset_from_feet(original, original_data)
        assert abs(pitched_offset) < abs(original_offset)

    for body_name, mass in (
        ("RightShin", 0.061),
        ("LeftShin", 0.061),
        ("RightFoot", 0.069),
        ("LeftFoot", 0.069),
    ):
        body_id = mujoco.mj_name2id(pitched, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert body_id >= 0
        assert pitched.body_mass[body_id] == pytest.approx(mass)

    foot_id = mujoco.mj_name2id(
        pitched, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision"
    )
    foot_rotation = pitched_data.geom_xmat[foot_id].reshape(3, 3)
    np.testing.assert_allclose(foot_rotation[:, 2], (0.0, 0.0, 1.0), atol=5e-5)
    np.testing.assert_allclose(pitched.geom_size[foot_id], (0.0275, 0.08, 0.01486))


def test_new_model_matches_original_outside_shin_and_foot() -> None:
    if not MODEL_PATH.is_file():
        pytest.skip("legacy LeggyBot model is not included in this repository")
    original = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    new = mujoco.MjModel.from_xml_path(str(NEW_MODEL_PATH))

    assert original.opt.timestep == new.opt.timestep
    assert original.opt.integrator == new.opt.integrator
    assert original.opt.solver == new.opt.solver
    assert original.opt.iterations == new.opt.iterations
    assert original.opt.ls_iterations == new.opt.ls_iterations
    np.testing.assert_array_equal(original.opt.gravity, new.opt.gravity)

    for name in ("BaseLink", "RightHip", "RightThigh", "LeftHip", "LeftThigh"):
        original_id = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_BODY, name)
        new_id = mujoco.mj_name2id(new, mujoco.mjtObj.mjOBJ_BODY, name)
        for field in (
            "body_pos",
            "body_quat",
            "body_ipos",
            "body_iquat",
            "body_mass",
            "body_inertia",
        ):
            np.testing.assert_array_equal(
                getattr(original, field)[original_id], getattr(new, field)[new_id]
            )

    for field in (
        "jnt_type",
        "jnt_axis",
        "jnt_range",
        "dof_damping",
        "dof_frictionloss",
        "dof_armature",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "actuator_gainprm",
        "actuator_biasprm",
    ):
        np.testing.assert_array_equal(getattr(original, field), getattr(new, field))

    for name in (
        "floor",
        "base_collision",
        "right_hip_collision",
        "right_thigh_collision",
        "left_hip_collision",
        "left_thigh_collision",
    ):
        original_id = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_GEOM, name)
        new_id = mujoco.mj_name2id(new, mujoco.mjtObj.mjOBJ_GEOM, name)
        for field in (
            "geom_type",
            "geom_pos",
            "geom_quat",
            "geom_size",
            "geom_friction",
            "geom_solref",
            "geom_condim",
            "geom_contype",
            "geom_conaffinity",
            "geom_group",
        ):
            np.testing.assert_array_equal(
                getattr(original, field)[original_id], getattr(new, field)[new_id]
            )

    assert original.nsensor == new.nsensor
    for sensor_id in range(original.nsensor):
        assert mujoco.mj_id2name(
            original, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id
        ) == mujoco.mj_id2name(new, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)

    visual_ids = np.flatnonzero(new.geom_group == 2)
    np.testing.assert_array_equal(new.geom_contype[visual_ids], 0)
    np.testing.assert_array_equal(new.geom_conaffinity[visual_ids], 0)
    assert new.body_mass.sum() == pytest.approx(1.314)
