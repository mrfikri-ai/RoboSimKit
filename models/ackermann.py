"""Ackermann (car-like) kinematic model using bicycle approximation.
"""

from __future__ import annotations

import numpy as np

from utils.angles import wrap_angle

def step(state, control, dt: float, *, L: float = 0.3, wrap: bool = False) -> np.ndarray:
    """Discrete-time Euler integration step for the bicycle model."""
    state = np.asarray(state, dtype=float).reshape(-1)
    control = np.asarray(control, dtype=float).reshape(-1)

    if state.size != 3:
        raise ValueError(f"state must be length 3 [x,y,theta], got {state.size}")
    if control.size != 2:
        raise ValueError(f"control must be length 2 [v,delta], got {control.size}")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if L <= 0:
        raise ValueError("wheelbase L must be > 0")

    x, y, theta = state
    v, delta = control

    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = (v / L) * np.tan(delta)

    next_state = np.array([
        x + dt * x_dot,
        y + dt * y_dot,
        theta + dt * theta_dot,
    ], dtype=float)

    if wrap:
        next_state[2] = wrap_angle(next_state[2])

    return next_state


__all__ = ["step", "wrap_angle"]
