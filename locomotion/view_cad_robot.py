"""Open or validate the first MuJoCo model exported from the custom CAD robot."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_MODEL = (
    Path(__file__).parent / "LeggyBot" / "mjcf" / "LeggyBot.xml"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run a dynamics check without opening a window.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=2.0,
        help="Headless simulation duration (default: 2 seconds).",
    )
    parser.add_argument(
        "--fixed-base",
        action="store_true",
        help="Weld the torso to the world and show joint axes for inspection.",
    )
    return parser


def _summary(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    stable = np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    return (
        f"model={model.names.decode().split(chr(0), 1)[0]!r}, "
        f"mass={model.body_mass.sum():.3f} kg, joints={model.nu}, "
        f"time={data.time:.3f} s, contacts={data.ncon}, "
        f"base_xyz={np.array2string(data.qpos[:3], precision=4)}, stable={stable}"
    )


def main() -> None:
    args = _parser().parse_args()
    model_path = args.model.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if args.fixed_base:
        weld_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, "inspection_base_weld"
        )
        if weld_id < 0:
            raise ValueError(
                f"{model_path} has no 'inspection_base_weld' equality constraint"
            )
        data.eq_active[weld_id] = True

    if args.headless:
        steps = max(0, round(args.seconds / model.opt.timestep))
        for _ in range(steps):
            mujoco.mj_step(model, data)
        print(_summary(model, data))
        return

    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (-0.02, 0.11, 0.25)
        viewer.cam.distance = 0.9
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -12
        if args.fixed_base:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        while viewer.is_running():
            start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
