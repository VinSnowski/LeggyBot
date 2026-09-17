import numpy as np

from locomotion.observation_latency import ObservationAgeModel


def test_device_delays_are_coherent_and_independent() -> None:
    age = ObservationAgeModel(hold_probability=1.0, seed=4)
    initial = tuple(np.full(2, value, dtype=np.float32) for value in range(4))
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(age.apply(*initial), initial, strict=True)
    )

    # Force only the servo bus to be one frame old. Position and velocity must
    # move together, while gyro and gravity remain current.
    age.servo._lag = 1
    age.imu._lag = 0
    current = tuple(value + 10.0 for value in initial)
    gyro, gravity, position, velocity = age.apply(*current)
    np.testing.assert_array_equal(gyro, current[0])
    np.testing.assert_array_equal(gravity, current[1])
    np.testing.assert_array_equal(position, initial[2])
    np.testing.assert_array_equal(velocity, initial[3])


def test_observation_age_reset_backfills_from_current_sample() -> None:
    age = ObservationAgeModel(hold_probability=1.0, seed=2)
    first = tuple(np.full(1, value, dtype=np.float32) for value in range(4))
    age.apply(*first)
    age.servo._lag = 1
    age.imu._lag = 1
    age.reset()
    after_reset = tuple(value + 20.0 for value in first)
    output = age.apply(*after_reset)
    for actual, expected in zip(output, after_reset, strict=True):
        np.testing.assert_array_equal(actual, expected)
