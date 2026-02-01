# library/models/unicycle.py
from __future__ import annotations

import numpy as np


def wrap_angle(theta: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def step(state, control, dt: float, *, wrap: bool = True, clip: bool = False,
         v_max: float | None = None, w_max: float | None = None) -> np.ndarray:
    """
    Unicycle kinematics (Euler integration).

    State:   [x, y, theta]
    Control: [v, w]  where v = forward linear speed, w = yaw rate

    x_{k+1} = x_k + v*cos(theta)*dt
    y_{k+1} = y_k + v*sin(theta)*dt
    th_{k+1}= th_k + w*dt
    """
    s = np.asarray(state, dtype=float).reshape(-1)
    u = np.asarray(control, dtype=float).reshape(-1)

    if s.size != 3:
        raise ValueError(f"state must be length 3 [x,y,theta], got {s.size}")
    if u.size != 2:
        raise ValueError(f"control must be length 2 [v,w], got {u.size}")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    x, y, th = s
    v, w = u

    if clip:
        if v_max is not None:
            v = float(np.clip(v, -v_max, v_max))
        if w_max is not None:
            w = float(np.clip(w, -w_max, w_max))

    x_next = x + v * np.cos(th) * dt
    y_next = y + v * np.sin(th) * dt
    th_next = th + w * dt

    if wrap:
        th_next = wrap_angle(th_next)

    return np.array([x_next, y_next, th_next], dtype=float)


class UnicycleModel:
    """
    Small convenience wrapper around step().
    Keeps parameters like limits and angle wrapping.
    """

    def __init__(self, *, wrap: bool = True, clip: bool = False,
                 v_max: float | None = None, w_max: float | None = None):
        self.wrap = wrap
        self.clip = clip
        self.v_max = v_max
        self.w_max = w_max

    def step(self, state, control, dt: float) -> np.ndarray:
        return step(
            state, control, dt,
            wrap=self.wrap, clip=self.clip, v_max=self.v_max, w_max=self.w_max
        )


__all__ = ["step", "UnicycleModel", "wrap_angle"]
