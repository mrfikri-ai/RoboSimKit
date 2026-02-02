from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from models.unicycle import step as unicycle_step, wrap_angle as wrap_u
from models.omni import step as omni_step, wrap_angle as wrap_o
from models.ackermann import step as ackermann_step, wrap_angle as wrap_a

from .go_to_goal import controller_pose_p


GoalFn = Callable[[float], np.ndarray]


def constant_goal(goal_state: np.ndarray) -> GoalFn:
    """Return a time-invariant goal function get_goal(t) -> goal_state."""

    goal_state = np.asarray(goal_state, dtype=float).reshape(3)

    def get_goal(_: float) -> np.ndarray:
        return goal_state

    return get_goal


def compute_go_to_goal_control(mode: str, goal_state: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute control input using the single proportional pose controller."""

    return controller_pose_p(mode, goal_state, state)


def select_model(mode: str, *, L_ack: float = 0.3):
    """Select (u_dim, step_fn, wrap_fn) for MODE."""

    if mode == "unicycle":
        return 2, unicycle_step, wrap_u
    if mode == "ackermann":
        return 2, (lambda s, u, ts: ackermann_step(s, u, ts, L=L_ack)), wrap_a
    if mode == "omnidirectional":
        return 3, omni_step, wrap_o

    raise ValueError("Invalid MODE. Expected 'unicycle', 'omnidirectional', or 'ackermann'.")


def plot_standard_results(
    *,
    mode: str,
    Ts: float,
    state_hist: np.ndarray,
    goal_hist: np.ndarray,
    u_hist: np.ndarray,
    goal_label: str = "goal",
) -> None:
    """Standard plots used by RoboSimKit examples."""

    t = np.arange(len(state_hist)) * Ts

    plt.figure()
    if mode == "unicycle":
        plt.plot(t, u_hist[:, 0], label="v [m/s]")
        plt.plot(t, u_hist[:, 1], label="omega [rad/s]")
    elif mode == "ackermann":
        plt.plot(t, u_hist[:, 0], label="v [m/s]")
        plt.plot(t, np.rad2deg(u_hist[:, 1]), label="delta [deg]")
    else:
        plt.plot(t, u_hist[:, 0], label="vx [m/s]")
        plt.plot(t, u_hist[:, 1], label="vy [m/s]")
        plt.plot(t, u_hist[:, 2], label="omega [rad/s]")
    plt.xlabel("t [s]")
    plt.ylabel("control input")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(t, state_hist[:, 0], label="px [m]")
    plt.plot(t, state_hist[:, 1], label="py [m]")
    plt.plot(t, state_hist[:, 2], label="theta [rad]")
    plt.plot(t, goal_hist[:, 0], ":", label=f"{goal_label} px")
    plt.plot(t, goal_hist[:, 1], ":", label=f"{goal_label} py")
    plt.plot(t, goal_hist[:, 2], ":", label=f"{goal_label} theta")
    plt.xlabel("t [s]")
    plt.ylabel("state")
    plt.grid(True)
    plt.legend()
