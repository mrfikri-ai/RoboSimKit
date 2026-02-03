# library/models/unicycle.py
import numpy as np

from utils.angles import wrap_angle

def B_unicycle(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta), 0.0],
        [np.sin(theta), 0.0],
        [0.0,           1.0],
    ], dtype=float)


def step(state, control, dt: float, *, wrap: bool = False) -> np.ndarray:
    # Update new state of the robot at time-step t+1
    # using discrete-time model of UNICYCLE model
    state = np.asarray(state, dtype=float).reshape(3,)
    control = np.asarray(control, dtype=float).reshape(2,)

    theta = state[2]
    B = B_unicycle(theta)

    next_state = state + dt * (B @ control)

    if wrap:
        next_state[2] = wrap_angle(next_state[2])

    return next_state


__all__ = ["step", "B_unicycle", "wrap_angle"]
