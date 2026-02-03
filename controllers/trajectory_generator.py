from __future__ import annotations

from typing import Callable

import numpy as np

from .go_to_goal import controller_pose_p


ReferenceFn = Callable[[float], np.ndarray]


def figure8_goal(t: float, *, A: float, B: float, w: float) -> np.ndarray:
    """Return a figure-8-like reference state [px, py, theta] at time t."""
    x = A * np.sin(w * t)
    y = 0.5 * B * np.sin(2.0 * w * t)

    x_d = A * w * np.cos(w * t)
    y_d = B * w * np.cos(2.0 * w * t)

    theta_d = np.arctan2(y_d, x_d)
    return np.array([x, y, theta_d], dtype=float)


def track_reference_step(
    mode: str,
    state: np.ndarray,
    t: float,
    *,
    get_reference: ReferenceFn,
    K_POS: float,
    K_THETA: float,
    L_ack: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (reference_state, u) at time t for arbitrary reference tracking.

    The reference is provided by get_reference(t) -> [px, py, theta].
    Control uses the single proportional pose controller.
    """

    reference_state = np.asarray(get_reference(float(t)), dtype=float).reshape(3)
    u = controller_pose_p(
        mode,
        reference_state,
        state,
        k_pos=K_POS,
        k_theta=K_THETA,
        L_ack=L_ack,
    )
    return reference_state, u


__all__ = [
    "ReferenceFn",
    "figure8_goal",
    "track_reference_step",
]
