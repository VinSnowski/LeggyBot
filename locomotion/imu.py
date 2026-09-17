"""ICM-42688-P measurement model and 6-axis gravity estimator."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


STANDARD_GRAVITY = 9.80665


@dataclass(frozen=True, slots=True)
class Icm42688Config:
    """Chosen register settings and estimator parameters for LeggyBot."""

    sample_hz: float = 50.0
    gyro_full_scale_rad_s: float = math.radians(500.0)
    accel_full_scale_m_s2: float = 16.0 * STANDARD_GRAVITY
    adc_counts: int = 65_536
    # TDK typical noise densities. Training adds vibration separately.
    gyro_noise_density_rad_s_sqrt_hz: float = math.radians(0.0028)
    accel_noise_density_m_s2_sqrt_hz: float = 70.0e-6 * STANDARD_GRAVITY
    gyro_bias_bound_rad_s: float = math.radians(0.5)
    accel_bias_bound_m_s2: float = 20.0e-3 * STANDARD_GRAVITY
    gyro_scale_error: float = 0.005
    accel_scale_error: float = 0.005
    cross_axis_error: float = 0.0125
    mounting_misalignment_rad: float = math.radians(1.0)
    gyro_vibration_std_rad_s: float = math.radians(0.5)
    accel_vibration_std_m_s2: float = 0.02 * STANDARD_GRAVITY
    gyro_bias_drift_rad_s_sqrt_s: float = math.radians(0.002)
    accel_bias_drift_m_s2_sqrt_s: float = 0.2e-3 * STANDARD_GRAVITY
    measurement_filter_alpha: float = 0.75
    gravity_correction_time_s: float = 0.25
    gyro_bias_learning_rate: float = 0.05
    accel_trust_full_error_g: float = 0.10
    accel_trust_zero_error_g: float = 0.35
    # Transport/sample age is modeled after fusion at the device level.
    output_delay_steps: int = 0

    @property
    def sample_period_s(self) -> float:
        return 1.0 / self.sample_hz

    @property
    def gyro_resolution_rad_s(self) -> float:
        return 2.0 * self.gyro_full_scale_rad_s / self.adc_counts

    @property
    def accel_resolution_m_s2(self) -> float:
        return 2.0 * self.accel_full_scale_m_s2 / self.adc_counts


ICM42688_CONFIG = Icm42688Config()


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-8:
        return fallback.copy()
    return vector / norm


def _quantize_and_clip(value: np.ndarray, full_scale: float, resolution: float) -> np.ndarray:
    clipped = np.clip(value, -full_scale, full_scale - resolution)
    return np.rint(clipped / resolution) * resolution


class Icm42688OrientationEstimator:
    """Causal roll/pitch estimator for a calibrated body-frame 6-axis IMU.

    A 6-axis IMU cannot observe absolute yaw. The filter therefore estimates
    only the gravity vector needed by the policy, using gyro propagation and a
    norm-gated accelerometer correction. Gyro bias components observable from
    gravity are learned slowly online.
    """

    def __init__(
        self,
        config: Icm42688Config = ICM42688_CONFIG,
        *,
        sensor_to_body: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.sensor_to_body = np.eye(3) if sensor_to_body is None else np.asarray(
            sensor_to_body, dtype=np.float64
        )
        if self.sensor_to_body.shape != (3, 3):
            raise ValueError("sensor_to_body must be a 3x3 matrix")
        self.reset()

    def reset(self) -> None:
        self._last_sample_time_ns: int | None = None
        self._gravity = np.array((0.0, 0.0, -1.0), dtype=np.float64)
        self._gyro_bias = np.zeros(3, dtype=np.float64)
        self._filtered_gyro: np.ndarray | None = None
        self._filtered_accel: np.ndarray | None = None
        self._delayed_gyro: list[np.ndarray] = []
        self._delayed_gravity: list[np.ndarray] = []
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.projected_gravity = self._gravity.astype(np.float32)

    def update(
        self,
        angular_velocity_sensor: np.ndarray,
        linear_acceleration_sensor: np.ndarray,
        sample_time_ns: int,
    ) -> bool:
        if sample_time_ns < 0:
            raise ValueError("IMU sample timestamp must be non-negative")
        if self._last_sample_time_ns is not None:
            elapsed = (sample_time_ns - self._last_sample_time_ns) / 1.0e9
            if elapsed < 0.75 * self.config.sample_period_s:
                return False
            dt = float(np.clip(
                elapsed,
                0.5 * self.config.sample_period_s,
                2.0 * self.config.sample_period_s,
            ))
        else:
            dt = self.config.sample_period_s

        gyro = self.sensor_to_body @ np.asarray(
            angular_velocity_sensor, dtype=np.float64
        )
        accel = self.sensor_to_body @ np.asarray(
            linear_acceleration_sensor, dtype=np.float64
        )
        gyro = _quantize_and_clip(
            gyro,
            self.config.gyro_full_scale_rad_s,
            self.config.gyro_resolution_rad_s,
        )
        accel = _quantize_and_clip(
            accel,
            self.config.accel_full_scale_m_s2,
            self.config.accel_resolution_m_s2,
        )
        alpha = self.config.measurement_filter_alpha
        if self._filtered_gyro is None:
            self._filtered_gyro = gyro.copy()
            self._filtered_accel = accel.copy()
            initial_norm_error = abs(float(np.linalg.norm(accel)) - STANDARD_GRAVITY)
            if initial_norm_error <= self.config.accel_trust_zero_error_g * STANDARD_GRAVITY:
                self._gravity = _normalize(-accel, self._gravity)
        else:
            self._filtered_gyro += alpha * (gyro - self._filtered_gyro)
            assert self._filtered_accel is not None
            self._filtered_accel += alpha * (accel - self._filtered_accel)

        assert self._filtered_accel is not None
        accel_norm = float(np.linalg.norm(self._filtered_accel))
        norm_error_g = abs(accel_norm - STANDARD_GRAVITY) / STANDARD_GRAVITY
        span = (
            self.config.accel_trust_zero_error_g
            - self.config.accel_trust_full_error_g
        )
        trust = float(np.clip(
            (self.config.accel_trust_zero_error_g - norm_error_g) / span,
            0.0,
            1.0,
        ))
        gravity_from_accel = _normalize(-self._filtered_accel, self._gravity)
        error = np.cross(self._gravity, gravity_from_accel)
        self._gyro_bias += (
            self.config.gyro_bias_learning_rate * trust * error * dt
        )
        corrected_gyro = self._filtered_gyro - self._gyro_bias
        predicted = _normalize(
            self._gravity - np.cross(corrected_gyro, self._gravity) * dt,
            self._gravity,
        )
        correction = trust * (1.0 - math.exp(-dt / self.config.gravity_correction_time_s))
        self._gravity = _normalize(
            (1.0 - correction) * predicted + correction * gravity_from_accel,
            predicted,
        )
        self._last_sample_time_ns = sample_time_ns

        self._delayed_gyro.append(corrected_gyro.copy())
        self._delayed_gravity.append(self._gravity.copy())
        max_length = self.config.output_delay_steps + 1
        if len(self._delayed_gyro) > max_length:
            self._delayed_gyro.pop(0)
            self._delayed_gravity.pop(0)
        output_index = max(0, len(self._delayed_gyro) - 1 - self.config.output_delay_steps)
        self.angular_velocity = self._delayed_gyro[output_index].astype(np.float32)
        self.projected_gravity = self._delayed_gravity[output_index].astype(np.float32)
        return True


def _normalize_torch(vector: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
    return torch.where(norm > 1.0e-8, vector / norm.clamp_min(1.0e-8), fallback)


class SimulatedIcm42688:
    """Vectorized ICM-42688-P measurement and fusion model for training."""

    def __init__(self, num_envs: int, device: str, config=ICM42688_CONFIG):
        self.num_envs = num_envs
        self.device = device
        self.config = config
        shape = (num_envs, 3)
        self.last_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.gravity = torch.tensor((0.0, 0.0, -1.0), device=device).repeat(num_envs, 1)
        self.gyro_bias_estimate = torch.zeros(shape, device=device)
        self.filtered_gyro = torch.zeros(shape, device=device)
        self.filtered_accel = torch.zeros(shape, device=device)
        self.output_gyro = torch.zeros(shape, device=device)
        self.output_gravity = self.gravity.clone()
        self.delay_gyro = torch.zeros(shape, device=device)
        self.delay_gravity = self.gravity.clone()
        self.observation_lag = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.fixed_gyro_bias = torch.zeros(shape, device=device)
        self.fixed_accel_bias = torch.zeros(shape, device=device)
        self.gyro_drift = torch.zeros(shape, device=device)
        self.accel_drift = torch.zeros(shape, device=device)
        self.gyro_scale = torch.ones(shape, device=device)
        self.accel_scale = torch.ones(shape, device=device)
        self.cross_axis = torch.zeros((num_envs, 3, 3), device=device)
        self.misalignment = torch.zeros(shape, device=device)
        self.randomization_strength = torch.zeros((num_envs, 1), device=device)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.last_step[ids] = -1
        self.gravity[ids] = torch.tensor((0.0, 0.0, -1.0), device=self.device)
        self.gyro_bias_estimate[ids] = 0.0
        self.gyro_drift[ids] = 0.0
        self.accel_drift[ids] = 0.0

    def _randomize(self, ids: torch.Tensor, strength: float) -> None:
        cfg = self.config
        count = ids.numel()
        s = float(strength)
        self.randomization_strength[ids] = s
        uniform = lambda shape: 2.0 * torch.rand(shape, device=self.device) - 1.0
        self.fixed_gyro_bias[ids] = uniform((count, 3)) * cfg.gyro_bias_bound_rad_s * s
        self.fixed_accel_bias[ids] = uniform((count, 3)) * cfg.accel_bias_bound_m_s2 * s
        self.gyro_scale[ids] = 1.0 + uniform((count, 3)) * cfg.gyro_scale_error * s
        self.accel_scale[ids] = 1.0 + uniform((count, 3)) * cfg.accel_scale_error * s
        cross = uniform((count, 3, 3)) * cfg.cross_axis_error * s
        diagonal = torch.arange(3, device=self.device)
        cross[:, diagonal, diagonal] = 0.0
        self.cross_axis[ids] = cross
        self.misalignment[ids] = uniform((count, 3)) * cfg.mounting_misalignment_rad * s
        self.observation_lag[ids] = torch.rand(count, device=self.device) < (0.5 * s)

    def _measure(
        self, gyro_true: torch.Tensor, accel_true: torch.Tensor, update: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        dt = cfg.sample_period_s
        ids = torch.where(update)[0]
        strength = self.randomization_strength[ids]
        self.gyro_drift[ids] += (
            torch.randn_like(self.gyro_drift[ids])
            * cfg.gyro_bias_drift_rad_s_sqrt_s
            * math.sqrt(dt)
            * strength
        )
        self.accel_drift[ids] += (
            torch.randn_like(self.accel_drift[ids])
            * cfg.accel_bias_drift_m_s2_sqrt_s
            * math.sqrt(dt)
            * strength
        )
        gyro = gyro_true[ids] * self.gyro_scale[ids]
        accel = accel_true[ids] * self.accel_scale[ids]
        gyro += torch.bmm(self.cross_axis[ids], gyro.unsqueeze(-1)).squeeze(-1)
        accel += torch.bmm(self.cross_axis[ids], accel.unsqueeze(-1)).squeeze(-1)
        gyro += torch.linalg.cross(self.misalignment[ids], gyro)
        accel += torch.linalg.cross(self.misalignment[ids], accel)
        gyro += self.fixed_gyro_bias[ids] + self.gyro_drift[ids]
        accel += self.fixed_accel_bias[ids] + self.accel_drift[ids]
        noise_bandwidth = cfg.sample_hz / 2.0
        gyro_std = (
            cfg.gyro_noise_density_rad_s_sqrt_hz * math.sqrt(noise_bandwidth)
            + cfg.gyro_vibration_std_rad_s * strength
        )
        accel_std = (
            cfg.accel_noise_density_m_s2_sqrt_hz * math.sqrt(noise_bandwidth)
            + cfg.accel_vibration_std_m_s2 * strength
        )
        gyro += torch.randn_like(gyro) * gyro_std
        accel += torch.randn_like(accel) * accel_std
        gyro = torch.clamp(
            torch.round(gyro / cfg.gyro_resolution_rad_s) * cfg.gyro_resolution_rad_s,
            -cfg.gyro_full_scale_rad_s,
            cfg.gyro_full_scale_rad_s - cfg.gyro_resolution_rad_s,
        )
        accel = torch.clamp(
            torch.round(accel / cfg.accel_resolution_m_s2) * cfg.accel_resolution_m_s2,
            -cfg.accel_full_scale_m_s2,
            cfg.accel_full_scale_m_s2 - cfg.accel_resolution_m_s2,
        )
        return gyro, accel

    def update(
        self,
        gyro_true: torch.Tensor,
        accel_true: torch.Tensor,
        episode_step: torch.Tensor,
        randomization_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update = episode_step != self.last_step
        fresh = update & ((self.last_step < 0) | (episode_step < self.last_step))
        if torch.any(fresh):
            self._randomize(torch.where(fresh)[0], randomization_scale)
        if not torch.any(update):
            return self.output_gyro, self.output_gravity

        ids = torch.where(update)[0]
        gyro, accel = self._measure(gyro_true, accel_true, update)
        alpha = self.config.measurement_filter_alpha
        continuing_ids = torch.where(update & ~fresh)[0]
        fresh_local = fresh[ids]
        if torch.any(fresh_local):
            fresh_ids = ids[fresh_local]
            self.filtered_gyro[fresh_ids] = gyro[fresh_local]
            self.filtered_accel[fresh_ids] = accel[fresh_local]
            fallback = self.gravity[fresh_ids]
            initial_norm_error = torch.abs(
                torch.linalg.vector_norm(accel[fresh_local], dim=1) - STANDARD_GRAVITY
            )
            initial_trusted = (
                initial_norm_error
                <= self.config.accel_trust_zero_error_g * STANDARD_GRAVITY
            )
            initialized = _normalize_torch(-accel[fresh_local], fallback)
            self.gravity[fresh_ids] = torch.where(
                initial_trusted[:, None], initialized, fallback
            )
            self.gyro_bias_estimate[fresh_ids] = 0.0
        if continuing_ids.numel():
            continuing_local = ~fresh_local
            self.filtered_gyro[continuing_ids] += alpha * (
                gyro[continuing_local] - self.filtered_gyro[continuing_ids]
            )
            self.filtered_accel[continuing_ids] += alpha * (
                accel[continuing_local] - self.filtered_accel[continuing_ids]
            )

        accel_norm = torch.linalg.vector_norm(self.filtered_accel[ids], dim=1)
        norm_error_g = torch.abs(accel_norm - STANDARD_GRAVITY) / STANDARD_GRAVITY
        span = self.config.accel_trust_zero_error_g - self.config.accel_trust_full_error_g
        trust = torch.clamp(
            (self.config.accel_trust_zero_error_g - norm_error_g) / span, 0.0, 1.0
        )
        gravity_accel = _normalize_torch(-self.filtered_accel[ids], self.gravity[ids])
        error = torch.linalg.cross(self.gravity[ids], gravity_accel)
        self.gyro_bias_estimate[ids] += (
            self.config.gyro_bias_learning_rate
            * trust[:, None]
            * error
            * self.config.sample_period_s
        )
        corrected_gyro = self.filtered_gyro[ids] - self.gyro_bias_estimate[ids]
        predicted = _normalize_torch(
            self.gravity[ids]
            - torch.linalg.cross(corrected_gyro, self.gravity[ids])
            * self.config.sample_period_s,
            self.gravity[ids],
        )
        correction = trust * (
            1.0
            - math.exp(
                -self.config.sample_period_s / self.config.gravity_correction_time_s
            )
        )
        self.gravity[ids] = _normalize_torch(
            (1.0 - correction[:, None]) * predicted
            + correction[:, None] * gravity_accel,
            predicted,
        )

        continuing_local = ~fresh[ids]
        if torch.any(continuing_local):
            change = torch.rand(ids.numel(), device=self.device) > 0.9
            resample = change & continuing_local
            if torch.any(resample):
                probability = 0.5 * self.randomization_strength[ids[resample], 0]
                self.observation_lag[ids[resample]] = (
                    torch.rand(
                        int(resample.sum().item()), device=self.device
                    ) < probability
                )
        use_previous = self.observation_lag[ids] & ~fresh[ids]
        self.output_gyro[ids] = torch.where(
            use_previous[:, None], self.delay_gyro[ids], corrected_gyro
        )
        self.output_gravity[ids] = torch.where(
            use_previous[:, None], self.delay_gravity[ids], self.gravity[ids]
        )
        self.delay_gyro[ids] = corrected_gyro
        self.delay_gravity[ids] = self.gravity[ids]
        self.last_step[ids] = episode_step[ids]
        return self.output_gyro, self.output_gravity
