# library/models/omni.py
from __future__ import annotations
import numpy as np

def wrap_angle(theta: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

def step(state, control, dt: float, wrap: bool = True) -> np.ndarray:
    """
    Omnidirectional model as a single integrator.
    """
    state = np.asarray(state, dtype=float).reshape(-1)
    control = np.asarray(control, dtype=float).reshape(-1)

    if state.size != 3:
        raise ValueError(f"state must be length 3 [x,y,theta], got {state.size}")
    if control.size != 3:
        raise ValueError(f"control must be length 3 [vx,vy,w], got {control.size}")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    
    next_state = state + control * dt

    if wrap:
        next_state[2] = wrap_angle(next_state[2])

    return next_state

__all__ = ["step", "wrap_angle"]