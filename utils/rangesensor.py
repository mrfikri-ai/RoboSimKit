"""2D range sensor helpers.

These utilities are model-agnostic: they operate on a robot pose [px, py, theta]
(in world frame) and obstacles described as polylines/polygons in world frame.

Indexing convention:
- The returned scan covers [0, 2*pi) with uniform angular steps.
- Index 0 points forward in the robot frame.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _iter_segments(polyline_xy: np.ndarray) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    v = np.asarray(polyline_xy, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError("Expected vertices as array of shape (N, 2).")
    if v.shape[0] < 2:
        return

    if not np.allclose(v[0], v[-1]):
        v = np.vstack([v, v[0]])

    for i in range(v.shape[0] - 1):
        yield v[i], v[i + 1]


def get_range_scan(
    state: np.ndarray,
    obstacles,
    *,
    sensing_range: float = 1.0,
    sensor_resolution: float = np.pi / 8,
) -> np.ndarray:
    """Simulate a 2D 360° range scan around the robot.

    Args:
      state: [px, py, theta]
      obstacles: either a single polyline/polygon (N,2) or a list of them.
      sensing_range: max range (meters)
      sensor_resolution: angular step (radians). The scan covers [0, 2*pi).

    Returns:
      distances: array of shape (N_sensors,) with values in [0, sensing_range]
                 where index 0 points forward in the robot frame.
    """

    if sensing_range <= 0:
        raise ValueError("sensing_range must be > 0")
    if sensor_resolution <= 0:
        raise ValueError("sensor_resolution must be > 0")

    s = np.asarray(state, dtype=float).reshape(3)
    origin = s[:2]
    theta = float(s[2])

    n_sensors = int(round(2.0 * np.pi / sensor_resolution))
    if n_sensors <= 0:
        raise ValueError("Invalid sensor_resolution; produced 0 sensors")

    sensor_angles = np.arange(n_sensors, dtype=float) * (2.0 * np.pi / n_sensors)
    world_angles = theta + sensor_angles

    if isinstance(obstacles, (list, tuple)):
        obstacle_list = list(obstacles)
    else:
        obstacle_list = [obstacles]

    distances = np.full(n_sensors, float(sensing_range), dtype=float)

    eps = 1e-12
    for i in range(n_sensors):
        r = np.array([np.cos(world_angles[i]), np.sin(world_angles[i])], dtype=float)
        best_t = float(sensing_range)

        for obst in obstacle_list:
            for q0, q1 in _iter_segments(obst):
                q = q0
                seg = q1 - q0
                rxs = _cross2(r, seg)
                if abs(rxs) < eps:
                    continue

                qp = q - origin
                t = _cross2(qp, seg) / rxs
                if t < 0.0 or t > best_t:
                    continue

                u = _cross2(qp, r) / rxs
                if 0.0 <= u <= 1.0:
                    best_t = float(t)

        distances[i] = best_t

    return distances


def compute_sensor_endpoints(
    state: np.ndarray,
    sensor_distances: np.ndarray,
    *,
    sensor_resolution: float = np.pi / 8,
) -> np.ndarray:
    """Convert range readings to world-frame endpoints.

    Returns:
      endpoints_xy: array of shape (N_sensors, 2)
    """

    s = np.asarray(state, dtype=float).reshape(3)
    d = np.asarray(sensor_distances, dtype=float).reshape(-1)
    if d.size == 0:
        return np.zeros((0, 2), dtype=float)

    n_sensors = d.size
    expected = int(round(2.0 * np.pi / sensor_resolution))
    if expected != n_sensors:
        raise ValueError(
            "sensor_distances length does not match sensor_resolution. "
            f"Expected {expected} readings, got {n_sensors}."
        )

    sensor_angles = np.arange(n_sensors, dtype=float) * (2.0 * np.pi / n_sensors)
    world_angles = float(s[2]) + sensor_angles

    return np.column_stack(
        [
            s[0] + d * np.cos(world_angles),
            s[1] + d * np.sin(world_angles),
        ]
    )


__all__ = ["get_range_scan", "compute_sensor_endpoints"]
