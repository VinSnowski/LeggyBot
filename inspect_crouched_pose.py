"""Visually inspect a contact-consistent, mildly crouched LeggyBot stance.

This is a geometry/posture inspection utility, not a hardware-feasibility
test.  It intentionally makes the position actuators very stiff and removes
their force limits so that the viewer holds the requested pose while the
floating base remains free.  Knee flexion is counter-rotated at the thighs so
the fixed-ankle soles retain their nominal, flat support orientation.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from locomotion.robot_profiles import LEGGYBOT_FIXED_ANKLE_PROFILE, make_native_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--knee-flex-deg",
        type=float,
        default=15.0,
        help="positive flexion magnitude applied to both knees (default: 15 deg)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=120.0,
        help="visualization-only position gain in N m/rad (default: 120)",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=8.0,
        help="visualization-only velocity damping in N m s/rad (default: 8)",
    )
    parser.add_argument(
        "--clearance-mm",
        type=float,
        default=0.5,
        help="initial sole clearance above the detected floor contact (default: 0.5)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="settle for --seconds and print the measured pose without opening a viewer",
    )
    parser.add_argument(
        "--start-running",
        action="store_true",
        help="start physics immediately instead of pausing on the requested inspection pose",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=2.0,
        help="headless settling duration (default: 2 s)",
    )
    return parser


def _joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    profile = LEGGYBOT_FIXED_ANKLE_PROFILE
    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in profile.joint_names
        ],
        dtype=np.int32,
    )
    if np.any(joint_ids < 0):
        raise ValueError("LeggyBot model is missing an expected policy joint")
    return model.jnt_qposadr[joint_ids], joint_ids


def _actuator_ids(model: mujoco.MjModel) -> np.ndarray:
    actuator_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in LEGGYBOT_FIXED_ANKLE_PROFILE.actuator_names
        ],
        dtype=np.int32,
    )
    if np.any(actuator_ids < 0):
        raise ValueError("LeggyBot model is missing an expected policy actuator")
    return actuator_ids


def crouched_joint_targets(knee_flex_deg: float) -> np.ndarray:
    """Return a fixed-ankle crouch with both sole support planes preserved.

    The exported knee convention uses negative angles for flexion.  Hip-roll
    remains nominal.  Each thigh counter-rotates by the same amount as its
    knee flexion, maintaining the nominal thigh-plus-knee angle and therefore
    the fixed sole's world pitch.
    """
    if not 0.0 < knee_flex_deg < 90.0:
        raise ValueError("--knee-flex-deg must be between 0 and 90")
    target = LEGGYBOT_FIXED_ANKLE_PROFILE.array("default_joint_pos")
    flexion = -np.deg2rad(knee_flex_deg)
    # Keep thigh + knee equal to the nominal value.  A knee-only flexion
    # pitches the fixed foot onto an edge, so it cannot represent a full-sole
    # double-support rest pose.
    target[1] -= flexion
    target[4] -= flexion
    target[2] = flexion
    target[5] = flexion
    lower = LEGGYBOT_FIXED_ANKLE_PROFILE.array("joint_lower")
    upper = LEGGYBOT_FIXED_ANKLE_PROFILE.array("joint_upper")
    if np.any(target < lower) or np.any(target > upper):
        raise ValueError("requested crouch is outside a knee joint limit")
    return target


def _has_terrain_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    terrain_geom_id: int,
    support_geom_ids: set[int],
) -> bool:
    supported: set[int] = set()
    for contact in data.contact[: data.ncon]:
        if contact.geom1 == terrain_geom_id:
            other = contact.geom2
        elif contact.geom2 == terrain_geom_id:
            other = contact.geom1
        else:
            continue
        if other in support_geom_ids:
            supported.add(other)
    return supported == support_geom_ids


def solve_contact_base_height(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_qpos_addresses: np.ndarray,
    target: np.ndarray,
    *,
    clearance_m: float,
) -> float:
    """Find the free-base Z at which the crouched feet first touch terrain."""
    if clearance_m < 0.0:
        raise ValueError("clearance must be non-negative")
    terrain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    if terrain_id < 0:
        raise ValueError("native model has no terrain geometry")
    support_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("right_foot_collision", "left_foot_collision")
    }
    if -1 in support_geom_ids:
        raise ValueError("native model is missing a foot support collision geometry")

    profile = LEGGYBOT_FIXED_ANKLE_PROFILE

    def set_pose(base_z: float) -> bool:
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = (0.0, 0.0, base_z)
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        data.qpos[joint_qpos_addresses] = target
        mujoco.mj_forward(model, data)
        return _has_terrain_contact(model, data, terrain_id, support_geom_ids)

    nominal_z = profile.initial_root_pos[2]
    upper = nominal_z + 0.50
    lower = nominal_z - 0.50
    while set_pose(upper):
        upper += 0.50
        if upper > 5.0:
            raise RuntimeError("could not find a terrain-clear upper base height")
    while not set_pose(lower):
        lower -= 0.50
        if lower < -5.0:
            raise RuntimeError("could not find a terrain-contact lower base height")

    # `lower` has terrain contact through *both* foot support geoms and
    # `upper` does not. The transition gives a configuration-consistent
    # double-support height without relying on visual mesh bounds.
    for _ in range(50):
        midpoint = 0.5 * (lower + upper)
        if set_pose(midpoint):
            lower = midpoint
        else:
            upper = midpoint
    return upper + clearance_m


def configure_visual_hold(
    model: mujoco.MjModel, actuator_ids: np.ndarray, *, kp: float, kd: float) -> None:
    """Make MuJoCo's position actuators stiff enough for posture inspection."""
    if kp <= 0.0 or kd < 0.0:
        raise ValueError("--kp must be positive and --kd must be non-negative")
    # Position actuators use gain * (ctrl - qpos) minus kv * qvel.  The CAD
    # actuator's physical force limit is intentionally disabled here: this is
    # a static geometry inspection, not a motor-capability validation.
    model.actuator_gainprm[actuator_ids, 0] = kp
    model.actuator_biasprm[actuator_ids, 1] = -kp
    model.actuator_biasprm[actuator_ids, 2] = -kd
    model.actuator_forcelimited[actuator_ids] = 0


def reset_crouched_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_qpos_addresses: np.ndarray,
    actuator_ids: np.ndarray,
    target: np.ndarray,
    base_z: float,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = (0.0, 0.0, base_z)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[joint_qpos_addresses] = target
    data.ctrl[actuator_ids] = target
    mujoco.mj_forward(model, data)


def _summary(data: mujoco.MjData, joint_qpos_addresses: np.ndarray) -> str:
    joint_degrees = np.rad2deg(data.qpos[joint_qpos_addresses])
    return (
        f"time={data.time:.3f}s, base_xyz={np.array2string(data.qpos[:3], precision=5)}, "
        f"joint_deg={np.array2string(joint_degrees, precision=2)}, contacts={data.ncon}"
    )


def main() -> None:
    args = _parser().parse_args()
    model = make_native_model(LEGGYBOT_FIXED_ANKLE_PROFILE)
    data = mujoco.MjData(model)
    joint_qpos_addresses, _ = _joint_addresses(model)
    actuator_ids = _actuator_ids(model)
    target = crouched_joint_targets(args.knee_flex_deg)
    base_z = solve_contact_base_height(
        model,
        data,
        joint_qpos_addresses,
        target,
        clearance_m=args.clearance_mm / 1_000.0,
    )
    configure_visual_hold(model, actuator_ids, kp=args.kp, kd=args.kd)
    reset_crouched_pose(
        model, data, joint_qpos_addresses, actuator_ids, target, base_z
    )

    print(
        "Crouched-pose inspection (unlimited high-gain hold; not actuator-feasible):\n"
        f"  requested knee flexion: {args.knee_flex_deg:.1f} deg\n"
        f"  contact-consistent initial base Z: {base_z:.5f} m\n"
        f"  target joints [right roll, thigh, knee, left roll, thigh, knee] deg: "
        f"{np.array2string(np.rad2deg(target), precision=2)}\n"
        "  Viewer: starts paused. Space runs/pauses physics; R resets the pose; "
        "use MuJoCo mouse controls to inspect."
    )

    if args.headless:
        steps = max(1, round(args.seconds / model.opt.timestep))
        for _ in range(steps):
            data.ctrl[actuator_ids] = target
            mujoco.mj_step(model, data)
        print(_summary(data, joint_qpos_addresses))
        return

    reset_requested = False
    running = args.start_running

    def key_callback(key: int) -> None:
        nonlocal reset_requested, running
        if key in (ord("R"), ord("r")):
            reset_requested = True
        elif key == 32:  # GLFW spacebar
            running = not running

    from mujoco import viewer

    with viewer.launch_passive(model, data, key_callback=key_callback) as window:
        window.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        window.cam.trackbodyid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, LEGGYBOT_FIXED_ANKLE_PROFILE.root_body
        )
        window.cam.distance = 0.9
        window.cam.azimuth = 135.0
        window.cam.elevation = -12.0
        while window.is_running():
            start = time.perf_counter()
            if reset_requested:
                reset_crouched_pose(
                    model, data, joint_qpos_addresses, actuator_ids, target, base_z
                )
                reset_requested = False
            if running:
                data.ctrl[actuator_ids] = target
                mujoco.mj_step(model, data)
            window.sync()
            remaining = model.opt.timestep - (time.perf_counter() - start)
            if remaining > 0.0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
