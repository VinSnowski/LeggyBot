"""Contract and validation helpers for adding a custom simulated robot.

This module does not invent physical parameters.  Sim-to-real requires measured
values from the actual mechanism; placeholders that merely compile are dangerous.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CustomRobotContract:
    mjcf_path: Path
    root_body: str
    foot_sites: tuple[str, ...]
    foot_geoms: tuple[str, ...]
    default_joint_positions: dict[str, float]
    action_scale: dict[str, float]
    control_frequency_hz: float = 50.0

    def validate_static(self) -> None:
        if not self.mjcf_path.is_file():
            raise FileNotFoundError(self.mjcf_path)
        if len(self.foot_sites) not in (2, 4):
            raise ValueError("A biped needs 2 feet and a quadruped needs 4 feet")
        if len(self.foot_sites) != len(self.foot_geoms):
            raise ValueError("foot_sites and foot_geoms must have equal length")
        if not self.default_joint_positions:
            raise ValueError("Measure and provide a stable default joint pose")
        if not self.action_scale or any(scale <= 0 for scale in self.action_scale.values()):
            raise ValueError("Every action scale must be positive")
        if self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive")


CUSTOM_ROBOT_CHECKLIST = (
    "MJCF has one floating root and collision-only foot geometry",
    "mass, inertia and center of mass were measured or identified",
    "position actuator gains reproduce the physical low-level controller",
    "effort, velocity and joint limits match firmware limits",
    "IMU angular velocity and gravity conventions match deployment",
    "joint ordering and sign conventions are identical end-to-end",
    "observation normalization and previous-action state are deployed",
    "a hardware safety supervisor can override policy commands",
)
