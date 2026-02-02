from __future__ import annotations

import numpy as np

from models.ackermann import wrap_angle as wrap_a
from models.omni import wrap_angle as wrap_o
from models.unicycle import wrap_angle as wrap_u


# Default controller gains (override by passing parameters).
ACKERMANN_K_V = 1.0
ACKERMANN_K_DELTA = 1.5

UNICYCLE_K_RHO = 1.2
UNICYCLE_K_ALPHA = 3.0

OMNI_K_P = 1.5
OMNI_K_TH = 2.0


def controller_ackermann(
    goal,
    state,
    *,
    k_v: float = ACKERMANN_K_V,
    k_delta: float = ACKERMANN_K_DELTA,
) -> np.ndarray:
    """control = [v, delta] (velocity, steering angle).

    Simple go-to-goal controller:
    - v drives forward proportional to distance
    - delta steers toward the goal direction
    """
    x, y, th = state
    gx, gy, _ = goal

    dx = gx - x
    dy = gy - y
    dist = np.hypot(dx, dy)

    ang_to_goal = np.arctan2(dy, dx)
    alpha = wrap_a(ang_to_goal - th)

    # No input saturation here by design.
    v = k_v * dist
    delta = k_delta * alpha

    return np.array([v, delta], dtype=float)


def controller_unicycle(
    goal,
    state,
    *,
    k_rho: float = UNICYCLE_K_RHO,
    k_alpha: float = UNICYCLE_K_ALPHA,
) -> np.ndarray:
    """control = [v, omega]."""
    x, y, th = state
    gx, gy, _ = goal

    dx = gx - x
    dy = gy - y
    rho = np.hypot(dx, dy)
    ang_to_goal = np.arctan2(dy, dx)
    alpha = wrap_u(ang_to_goal - th)

    # No input saturation here by design.
    v = k_rho * rho

    # Smoothly gate omega to 0 as rho -> 0 (avoids a hard stop `if`).
    eps = 1e-9
    omega = k_alpha * alpha * (rho / (rho + eps))

    return np.array([v, omega], dtype=float)


def controller_omni(
    goal,
    state,
    *,
    k_p: float = OMNI_K_P,
    k_th: float = OMNI_K_TH,
) -> np.ndarray:
    """control = [vx, vy, omega] in WORLD frame (matches omni model)."""
    x, y, th = state
    gx, gy, gth = goal

    ex = gx - x
    ey = gy - y
    eth = wrap_o(gth - th)

    vx = k_p * ex
    vy = k_p * ey
    omega = k_th * eth

    return np.array([vx, vy, omega], dtype=float)


__all__ = [
    "ACKERMANN_K_V",
    "ACKERMANN_K_DELTA",
    "UNICYCLE_K_RHO",
    "UNICYCLE_K_ALPHA",
    "OMNI_K_P",
    "OMNI_K_TH",
    "controller_ackermann",
    "controller_unicycle",
    "controller_omni",
]
