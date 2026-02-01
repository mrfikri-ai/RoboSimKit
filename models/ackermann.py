"""Ackermann (car-like) kinematic model using bicycle approximation.

State:   [x, y, theta]
Control: [v, delta] where delta is the steering angle (front wheel)

Continuous-time model:
  x_dot     = v * cos(theta)
  y_dot     = v * sin(theta)
  theta_dot = (v / L) * tan(delta)

This module intentionally does NOT clip/saturate inputs. If you want steering
limits, do it in the controller.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(theta: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def step(state, control, dt: float, *, L: float = 0.3, wrap: bool = True) -> np.ndarray:
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
