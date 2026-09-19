from pathlib import Path

import pytest

from locomotion.evaluate_omni_run import (
    Objectives,
    checkpoint_iteration,
    commands_for_task,
    discover_checkpoints,
    judge_mode,
)
from locomotion.tasks import WALKING_TASK_ID


def moving_metrics(**overrides):
    result = {
        "survival": 1.0,
        "linear_error_mps": .02,
        "yaw_error_radps": .05,
        "path_rmse_m": .10,
        "heading_rmse_rad": .15,
        "bilateral_step_fraction": 1.0,
        "mean_base_pitch_deg": 2.0,
        "rms_base_pitch_deg": 4.0,
        "lateral_foot_line_error_m": .01,
    }
    result.update(overrides)
    return result


def stop_metrics(**overrides):
    result = {
        "survival": 1.0,
        "planar_speed_mps": .005,
        "mean_velocity": [0., 0., .01],
        "joint_rmse_rad": .05,
        "foot_fore_aft_offset_m": .01,
        "maximum_mean_liftoffs_per_foot": 0.,
        "mean_base_pitch_deg": 2.0,
        "rms_base_pitch_deg": 3.0,
    }
    result.update(overrides)
    return result


def test_mode_judgement_requires_tracking_stability_and_real_steps():
    objectives = Objectives()
    assert judge_mode("forward", moving_metrics(), objectives)["passed"]
    failed = judge_mode(
        "left", moving_metrics(linear_error_mps=.031, bilateral_step_fraction=.75), objectives
    )
    assert not failed["passed"]
    assert not failed["checks"]["linear_error"]
    assert not failed["checks"]["step_fraction"]


def test_mode_judgement_rejects_sustained_or_oscillating_base_pitch():
    objectives = Objectives()
    biased = judge_mode("forward", moving_metrics(mean_base_pitch_deg=-12.0), objectives)
    oscillating = judge_mode("forward", moving_metrics(rms_base_pitch_deg=13.0), objectives)
    assert not biased["passed"]
    assert not biased["checks"]["base_pitch_bias"]
    assert not oscillating["passed"]
    assert not oscillating["checks"]["base_pitch_rms"]


def test_lateral_judgement_enforces_foot_line_error():
    objectives = Objectives()
    assert judge_mode(
        "left", moving_metrics(), objectives, include_lateral_foot_line=True
    )["passed"]
    failed = judge_mode(
        "right",
        moving_metrics(lateral_foot_line_error_m=.026),
        objectives,
        include_lateral_foot_line=True,
    )
    assert not failed["passed"]
    assert not failed["checks"]["lateral_foot_line"]


def test_stop_judgement_rejects_creep_scissor_pose_and_stepping():
    objectives = Objectives()
    assert judge_mode("stop", stop_metrics(), objectives)["passed"]
    failed = judge_mode(
        "stop",
        stop_metrics(
            planar_speed_mps=.02,
            foot_fore_aft_offset_m=.03,
            maximum_mean_liftoffs_per_foot=2.,
        ),
        objectives,
    )
    assert not failed["passed"]
    assert not failed["checks"]["stop_speed"]
    assert not failed["checks"]["foot_offset"]
    assert not failed["checks"]["stop_liftoffs"]


def test_checkpoint_discovery_is_numeric_and_always_includes_final(tmp_path: Path):
    for name in ("model_0.pt", "model_50.pt", "model_100.pt", "model_149.pt"):
        (tmp_path / name).touch()
    selected = discover_checkpoints(tmp_path, start=0, end=None, stride=100)
    assert [checkpoint_iteration(path) for path in selected] == [0, 100, 149]
    with pytest.raises(FileNotFoundError):
        discover_checkpoints(tmp_path, start=200, end=None, stride=50)


def test_evaluation_matches_the_trained_command_modes():
    assert tuple(commands_for_task(WALKING_TASK_ID)) == (
        "stop",
        "forward",
        "curve_left",
        "curve_right",
    )
