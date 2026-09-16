"""Open a raw SolidWorks-exported URDF for visual assembly inspection.

The SolidWorks exporter uses ROS ``package://`` mesh paths, which native
MuJoCo cannot resolve by itself.  This tool resolves them to the checked-in
package directory in a temporary copy.  Its default static mode also converts
all joints to fixed *only in that temporary preview*; it is useful when an
export has not yet received valid joint axes and limits.
"""

from __future__ import annotations

import argparse
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


CONTROL_RANGES = {
    "RightBaseToHip": (-0.174533, 0.785398),
    "RightHipToThigh": (-1.047198, 1.047198),
    "RightKnee": (-2.094395, 0.261799),
    "LeftBaseToHip": (-0.785398, 0.174533),
    "LeftHipToThigh": (-1.047198, 1.047198),
    "LeftKnee": (-2.094395, 0.261799),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path, help="path to the exported URDF")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="preserve joint types; use only after all joint axes and limits are valid",
    )
    parser.add_argument(
        "--fixed-base",
        action="store_true",
        help="weld the FloatingBase link to the world for leg inspection",
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="add position actuators for the movable leg joints",
    )
    parser.add_argument(
        "--control-kp",
        type=float,
        default=17.8,
        help="position actuator stiffness for --control (default: 17.8)",
    )
    parser.add_argument("--headless", action="store_true", help="compile and report only")
    parser.add_argument(
        "--image",
        type=Path,
        help="render a static PNG inspection image instead of opening a viewer",
    )
    return parser


def _configure_inspection_model(
    spec: mujoco.MjSpec, *, fixed_base: bool, control: bool, control_kp: float
) -> None:
    floating_base = next(
        (body for body in spec.bodies if body.name == "FloatingBase"), None
    )
    if floating_base is not None and not any(
        joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in spec.joints
    ):
        floating_base.add_freejoint(name="floating_base")

    capsule_names = {
        "right_thigh_collision",
        "right_shin_collision",
        "left_thigh_collision",
        "left_shin_collision",
    }
    for geom in spec.geoms:
        if geom.name in capsule_names:
            geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        if geom.contype == 0 and geom.conaffinity == 0:
            geom.group = 2
        else:
            geom.group = 3
            geom.condim = 4
            geom.friction[:] = (0.9, 0.02, 0.002)
            geom.solref[:] = (0.01, 1.0)

    for joint in spec.joints:
        if joint.name in CONTROL_RANGES:
            joint.armature = 0.028

    if control:
        if control_kp <= 0.0:
            raise ValueError("--control-kp must be positive")
        for joint in spec.joints:
            if joint.type not in {
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            }:
                continue
            actuator = spec.add_actuator()
            actuator.name = f"inspect_{joint.name}"
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.target = joint.name
            actuator.set_to_position(control_kp, kv=0.0)
            actuator.ctrlrange[:] = CONTROL_RANGES.get(joint.name, (-3.141593, 3.141593))
            actuator.ctrllimited = True
            actuator.forcerange[:] = (-2.942, 2.942)
            actuator.forcelimited = True

    if not any(geom.name == "ground" for geom in spec.geoms):
        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0.0, 0.0, 0.05),
            friction=(0.9, 0.02, 0.002),
            condim=4,
            solref=(0.01, 1.0),
            rgba=(0.18, 0.21, 0.24, 1.0),
            group=3,
        )

    if fixed_base:
        weld = spec.add_equality()
        weld.name = "inspection_base_weld"
        weld.type = mujoco.mjtEq.mjEQ_WELD
        weld.objtype = mujoco.mjtObj.mjOBJ_BODY
        weld.name1 = "FloatingBase"
        weld.active = True
        weld.data[:] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _package_root(urdf_path: Path, package_name: str) -> Path:
    for parent in (urdf_path.parent, *urdf_path.parents):
        if parent.name == package_name:
            return parent
    raise ValueError(
        f"could not locate package directory {package_name!r} above {urdf_path}"
    )


def _make_preview_urdf(urdf_path: Path, output_path: Path, *, dynamic: bool) -> int:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    invalid_revolute: list[str] = []

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename is None or not filename.startswith("package://"):
            continue
        package_and_relative = filename.removeprefix("package://").split("/", 1)
        if len(package_and_relative) != 2:
            raise ValueError(f"invalid package mesh path {filename!r}")
        package_name, relative_path = package_and_relative
        resolved = _package_root(urdf_path, package_name) / relative_path
        if not resolved.is_file():
            raise FileNotFoundError(f"URDF mesh path does not exist: {resolved}")
        mesh.set("filename", str(resolved))

    for joint in root.findall("joint"):
        joint_type = joint.get("type")
        axis = joint.find("axis")
        axis_xyz = np.fromstring(axis.get("xyz", ""), sep=" ") if axis is not None else np.array([])
        invalid_axis = axis_xyz.shape != (3,) or np.linalg.norm(axis_xyz) < 1.0e-8
        if joint_type in {"revolute", "continuous", "prismatic"} and invalid_axis:
            invalid_revolute.append(joint.get("name", "<unnamed>"))
            if not dynamic:
                joint.set("type", "fixed")
                axis = joint.find("axis")
                if axis is not None:
                    joint.remove(axis)
                limit = joint.find("limit")
                if limit is not None:
                    joint.remove(limit)

    if dynamic and invalid_revolute:
        names = ", ".join(invalid_revolute)
        raise ValueError(
            "cannot create a dynamic MuJoCo model: joints with zero/missing axes: " + names
        )
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return len(invalid_revolute)


def main() -> None:
    args = _parser().parse_args()
    urdf_path = args.urdf.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)

    with tempfile.TemporaryDirectory(prefix="leggybot-urdf-preview-") as directory:
        preview_path = Path(directory) / urdf_path.name
        invalid_count = _make_preview_urdf(
            urdf_path, preview_path, dynamic=args.dynamic
        )
        spec = mujoco.MjSpec.from_file(str(preview_path))
        _configure_inspection_model(
            spec,
            fixed_base=args.fixed_base,
            control=args.control,
            control_kp=args.control_kp,
        )
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        mode = "dynamic" if args.dynamic else "static"
        print(
            f"Loaded {urdf_path.name} in {mode} preview: "
            f"bodies={model.nbody}, joints={model.njnt}, actuators={model.nu}, "
            f"geoms={model.ngeom}, "
            f"mass={model.body_mass.sum():.3f} kg"
        )
        if invalid_count:
            print(
                f"WARNING: converted {invalid_count} zero-axis joints to fixed only "
                "for this visual preview. The source URDF remains unchanged."
            )
        if args.image is not None:
            image_path = args.image.expanduser().resolve()
            image_path.parent.mkdir(parents=True, exist_ok=True)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            # Static previews merge fixed URDF links into the world body, so
            # there may be no robot subtree COM. Geometry bounds work in both
            # static and dynamic imports.
            geom_positions = data.geom_xpos[: model.ngeom]
            camera.lookat[:] = 0.5 * (geom_positions.min(axis=0) + geom_positions.max(axis=0))
            camera.distance = max(1.2, 2.5 * float(model.geom_rbound.max()))
            camera.azimuth = 135.0
            camera.elevation = -10.0
            # MuJoCo's default offscreen framebuffer is 640x480 for imported
            # URDFs unless an MJCF visual/global clause enlarges it.
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                renderer.update_scene(data, camera=camera)
                Image.fromarray(renderer.render()).save(image_path)
            print(f"Rendered {image_path}")
            return
        if args.headless:
            return

        from mujoco import viewer

        with viewer.launch_passive(model, data) as window:
            root_body = 1 if model.nbody > 1 else 0
            window.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            window.cam.trackbodyid = root_body
            window.cam.distance = 1.2
            window.cam.azimuth = 135.0
            window.cam.elevation = -10.0
            window.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            next_step = time.perf_counter()
            next_render = next_step
            render_period = 1.0 / 60.0
            while window.is_running():
                now = time.perf_counter()
                steps = 0
                while now >= next_step and steps < 50:
                    mujoco.mj_step(model, data)
                    next_step += model.opt.timestep
                    steps += 1
                if steps == 50 and now >= next_step:
                    next_step = now + model.opt.timestep
                if now >= next_render:
                    window.sync()
                    next_render = now + render_period
                remaining = min(next_step, next_render) - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(min(remaining, model.opt.timestep))


if __name__ == "__main__":
    main()
