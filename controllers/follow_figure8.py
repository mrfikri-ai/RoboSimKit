from __future__ import annotations

import numpy as np

from utils.angles import wrap_angle

from .go_to_goal import controller_ackermann, controller_omni, controller_unicycle


def figure8_reference(
    t: float,
    *,
    A: float,
    B: float,
    w: float,
    mode: str,
    L_ack: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (desired_state, feedforward_u) for a figure-8 trajectory.

    desired_state: [x_d, y_d, theta_d]
    feedforward_u:
      - omni:      [vx_d, vy_d, omega_d]
      - unicycle:  [v_d, omega_d]
      - ackermann: [v_d, delta_d]

    Trajectory:
      x = A sin(w t)
      y = (B/2) sin(2 w t)
    """

    x = A * np.sin(w * t)
    y = 0.5 * B * np.sin(2.0 * w * t)

    x_d = A * w * np.cos(w * t)
    y_d = B * w * np.cos(2.0 * w * t)

    x_dd = -A * (w**2) * np.sin(w * t)
    y_dd = -2.0 * B * (w**2) * np.sin(2.0 * w * t)

    theta_d = np.arctan2(y_d, x_d)

    v = float(np.hypot(x_d, y_d))

    denom = (x_d * x_d + y_d * y_d) ** 1.5
    if denom == 0.0:
        kappa = 0.0
    else:
        kappa = float((x_d * y_dd - y_d * x_dd) / denom)

    omega = v * kappa

    desired_state = np.array([x, y, theta_d], dtype=float)

    if mode == "unicycle":
        u_ff = np.array([v, omega], dtype=float)
    elif mode == "ackermann":
        delta = float(np.arctan(L_ack * kappa))
        u_ff = np.array([v, delta], dtype=float)
    elif mode == "omnidirectional":
        u_ff = np.array([x_d, y_d, omega], dtype=float)
    else:
        raise ValueError("Invalid MODE. Expected 'unicycle', 'omnidirectional', or 'ackermann'.")

    return desired_state, u_ff


def figure8_goal(t: float, *, A: float, B: float, w: float) -> np.ndarray:
    """Return reference state [px, py, theta] at time t."""
    x = A * np.sin(w * t)
    y = 0.5 * B * np.sin(2.0 * w * t)

    x_d = A * w * np.cos(w * t)
    y_d = B * w * np.cos(2.0 * w * t)

    theta_d = np.arctan2(y_d, x_d)
    return np.array([x, y, theta_d], dtype=float)


def controller_follow_figure8(
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
) -> np.ndarray:
    """Tracking controller: feedforward + feedback based on go-to-goal controllers."""

    desired_state, u_ff = figure8_reference(t, A=A, B=B, w=w, mode=mode, L_ack=L_ack)

    if mode == "unicycle":
        u_fb = controller_unicycle(desired_state, state, k_rho=K_POS, k_alpha=K_THETA)
        return u_ff + u_fb

    if mode == "ackermann":
        u_fb = controller_ackermann(desired_state, state, k_v=K_POS, k_delta=K_THETA)
        return u_ff + u_fb

    if mode == "omnidirectional":
        u_fb = controller_omni(desired_state, state, k_p=K_POS, k_th=K_THETA)
        u = u_ff + u_fb
        u[2] = wrap_angle(float(u[2]))
        return u

    raise ValueError("Invalid MODE. Expected 'unicycle', 'omnidirectional', or 'ackermann'.")


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
    """Return (desired_state, u) at time t for figure-8 tracking."""

    desired_state = figure8_goal(t, A=A, B=B, w=w)
    u = controller_follow_figure8(
        mode,
        state,
        t,
        A=A,
        B=B,
        w=w,
        K_POS=K_POS,
        K_THETA=K_THETA,
        L_ack=L_ack,
    )
    return desired_state, u
