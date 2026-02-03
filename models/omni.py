# library/models/omni.py
from __future__ import annotations
import numpy as np

from utils.angles import wrap_angle

def step(state, control, dt: float, wrap: bool = False) -> np.ndarray:
    """
    Omnidirectional model as a single integrator.
    """
    state = np.asarray(state, dtype=float).reshape(-1)
    control = np.asarray(control, dtype=float).reshape(-1)

    next_state = state + control * dt

    if wrap:
        next_state[2] = wrap_angle(next_state[2])

    return next_state

__all__ = ["step", "wrap_angle"]