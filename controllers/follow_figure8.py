from __future__ import annotations

import numpy as np

from .go_to_goal import controller_pose_p


def figure8_goal(t: float, *, A: float, B: float, w: float) -> np.ndarray:
    """Return reference state [px, py, theta] at time t."""
    x = A * np.sin(w * t)
    y = 0.5 * B * np.sin(2.0 * w * t)

    x_d = A * w * np.cos(w * t)
    y_d = B * w * np.cos(2.0 * w * t)

    theta_d = np.arctan2(y_d, x_d)
    return np.array([x, y, theta_d], dtype=float)


def follow_figure8_step(
    mode: str,
    state: np.ndarray,
    t: float,
    *,
    A: float,
    B: float,
    w: float,
    K_POS: float,
    K_THETA: float,
    L_ack: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (desired_state, u) at time t for figure-8 tracking.

    Uses the single proportional pose controller (no separate tracking controller).
    """

    desired_state = figure8_goal(t, A=A, B=B, w=w)
    u = controller_pose_p(
        mode,
        desired_state,
        state,
        k_pos=K_POS,
        k_theta=K_THETA,
        L_ack=L_ack,
    )
    return desired_state, u
