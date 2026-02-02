from __future__ import annotations

import numpy as np

from utils.angles import wrap_angle


# Default controller gains (override by passing parameters).
ACKERMANN_K_V = 1.0
ACKERMANN_K_DELTA = 1.5

UNICYCLE_K_RHO = 1.2
UNICYCLE_K_ALPHA = 3.0

OMNI_K_P = 1.5
OMNI_K_TH = 2.0


def controller_pose_p(
    mode: str,
    goal,
    state,
    *,
    k_pos: float | None = None,
    k_theta: float | None = None,
    L_ack: float = 0.3,
) -> np.ndarray:
    """Single proportional pose controller.

    Given current pose state=[x, y, theta] and goal=[gx, gy, gtheta], compute a
    proportional control command for the selected kinematic model:

      - unicycle:        u=[v, omega]
      - ackermann:       u=[v, delta]
      - omnidirectional: u=[vx, vy, omega] (WORLD frame)

    Gains:
      - k_pos controls position error response
      - k_theta controls heading error response

    If gains are not provided, per-model defaults are used.
    """

    mode = str(mode)
    goal = np.asarray(goal, dtype=float).reshape(3)
    state = np.asarray(state, dtype=float).reshape(3)

    x, y, th = state
    gx, gy, gth = goal

    if mode == "unicycle":
        kp = UNICYCLE_K_RHO if k_pos is None else float(k_pos)
        kth = UNICYCLE_K_ALPHA if k_theta is None else float(k_theta)

        dx = gx - x
        dy = gy - y
        rho = float(np.hypot(dx, dy))
        ang_to_goal = float(np.arctan2(dy, dx))
        alpha = float(wrap_angle(ang_to_goal - th))

        v = kp * rho

        # Smoothly gate omega to 0 as rho -> 0 (avoids a hard stop `if`).
        eps = 1e-9
        omega = kth * alpha * (rho / (rho + eps))
        return np.array([v, omega], dtype=float)

    if mode == "ackermann":
        kp = ACKERMANN_K_V if k_pos is None else float(k_pos)
        kth = ACKERMANN_K_DELTA if k_theta is None else float(k_theta)

        dx = gx - x
        dy = gy - y
        dist = float(np.hypot(dx, dy))
        ang_to_goal = float(np.arctan2(dy, dx))
        alpha = float(wrap_angle(ang_to_goal - th))

        v = kp * dist
        delta = kth * alpha
        return np.array([v, delta], dtype=float)

    if mode in ("omnidirectional", "omni"):
        kp = OMNI_K_P if k_pos is None else float(k_pos)
        kth = OMNI_K_TH if k_theta is None else float(k_theta)

        ex = gx - x
        ey = gy - y
        eth = float(wrap_angle(gth - th))

        vx = kp * ex
        vy = kp * ey
        omega = kth * eth
        return np.array([vx, vy, omega], dtype=float)

    raise ValueError("Invalid MODE. Expected 'unicycle', 'omnidirectional', or 'ackermann'.")


def controller_ackermann(
    goal,
    state,
    *,
    k_v: float = ACKERMANN_K_V,
    k_delta: float = ACKERMANN_K_DELTA,
) -> np.ndarray:
    """control = [v, delta] (velocity, steering angle).

    Simple go-to-goal controller:
    """
    # Compatibility wrapper around controller_pose_p.
    return controller_pose_p(
        "ackermann",
        goal,
        state,
        k_pos=k_v,
        k_theta=k_delta,
    )


def controller_unicycle(
    goal,
    state,
    *,
    k_rho: float = UNICYCLE_K_RHO,
    k_alpha: float = UNICYCLE_K_ALPHA,
) -> np.ndarray:
    """control = [v, omega]."""
    # Compatibility wrapper around controller_pose_p.
    return controller_pose_p(
        "unicycle",
        goal,
        state,
        k_pos=k_rho,
        k_theta=k_alpha,
    )


def controller_omni(
    goal,
    state,
    *,
    k_p: float = OMNI_K_P,
    k_th: float = OMNI_K_TH,
) -> np.ndarray:
    """control = [vx, vy, omega] in WORLD frame (matches omni model)."""
    # Compatibility wrapper around controller_pose_p.
    return controller_pose_p(
        "omnidirectional",
        goal,
        state,
        k_pos=k_p,
        k_theta=k_th,
    )


__all__ = [
    "ACKERMANN_K_V",
    "ACKERMANN_K_DELTA",
    "UNICYCLE_K_RHO",
    "UNICYCLE_K_ALPHA",
    "OMNI_K_P",
    "OMNI_K_TH",
    "controller_pose_p",
    "controller_ackermann",
    "controller_unicycle",
    "controller_omni",
]
