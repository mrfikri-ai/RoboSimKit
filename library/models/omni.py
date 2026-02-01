# library/models/omni.py
from __future__ import annotations

import numpy as np


def wrap_angle(theta: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def step(state, control, dt: float, *, frame: str = "body",
         wrap: bool = True, clip: bool = False,
         v_max: float | None = None, w_max: float | None = None) -> np.ndarray:
    """
    Omnidirectional base kinematics (Euler integration).

    State:   [x, y, theta]
    Control: [vx, vy, w]
      - vx, vy: planar velocity
      - w: yaw rate

    frame="body": (default) vx,vy are in robot frame.
        [dx,dy]_world = R(theta) * [vx,vy]_body

    frame="world": vx,vy already in world frame.
    """
    s = np.asarray(state, dtype=float).reshape(-1)
    u = np.asarray(control, dtype=float).reshape(-1)

    if s.size != 3:
        raise ValueError(f"state must be length 3 [x,y,theta], got {s.size}")
    if u.size != 3:
        raise ValueError(f"control must be length 3 [vx,vy,w], got {u.size}")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    x, y, th = s
    vx, vy, w = u

    if clip:
        if v_max is not None:
            vx = float(np.clip(vx, -v_max, v_max))
            vy = float(np.clip(vy, -v_max, v_max))
        if w_max is not None:
            w = float(np.clip(w, -w_max, w_max))

    if frame not in ("body", "world"):
        raise ValueError("frame must be 'body' or 'world'")

    if frame == "body":
        c, s_ = np.cos(th), np.sin(th)
        vx_w = c * vx - s_ * vy
        vy_w = s_ * vx + c * vy
    else:
        vx_w, vy_w = vx, vy

    x_next = x + vx_w * dt
    y_next = y + vy_w * dt
    th_next = th + w * dt

    if wrap:
        th_next = wrap_angle(th_next)

    return np.array([x_next, y_next, th_next], dtype=float)


class OmniModel:
    """
    Convenience wrapper around step().
    """

    def __init__(self, *, frame: str = "body", wrap: bool = True, clip: bool = False,
                 v_max: float | None = None, w_max: float | None = None):
        self.frame = frame
        self.wrap = wrap
        self.clip = clip
        self.v_max = v_max
        self.w_max = w_max

    def step(self, state, control, dt: float) -> np.ndarray:
        return step(
            state, control, dt,
            frame=self.frame, wrap=self.wrap,
            clip=self.clip, v_max=self.v_max, w_max=self.w_max
        )


__all__ = ["step", "OmniModel", "wrap_angle"]
