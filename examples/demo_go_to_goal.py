import os
import sys

# Ensure repo root is on path so imports work when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from visualization.plotter2d import MobileRobotPlotter2D
from models.unicycle import step as unicycle_step, wrap_angle as wrap_u
from models.omni import step as omni_step, wrap_angle as wrap_o
from models.ackermann import step as ackermann_step, wrap_angle as wrap_a

# ----------------------------
# Choose robot type here
# ----------------------------
MODE = "omnidirectional"          # "unicycle", "omnidirectional", or "ackermann"
SHOW_2D = True

# Simulation settings
Ts = 0.01
t_max = 10.0

# Initial state and goal (state = [px, py, theta])
init_state = np.array([0.0, 0.0, np.pi/2])
goal_state = np.array([1.5, 1.0, 0.0])

# Visualization field limits
field_x = (-2.5, 2.5)
field_y = (-2.0, 2.0)


def controller_ackermann(goal, state):
    """control = [v, delta] (velocity, steering angle).

    Simple go-to-goal controller:
    - v drives forward proportional to distance
    - delta steers toward the goal direction
    """
    x, y, th = state
    gx, gy, _ = goal

    dx = gx - x
    dy = gy - y
    dist = np.hypot(dx, dy)

    ang_to_goal = np.arctan2(dy, dx)
    alpha = wrap_a(ang_to_goal - th)

    # Controller gains
    k_v = 1.0
    k_delta = 1.5

    # Forward speed (cap to keep simulation stable)
    v = min(k_v * dist, 1.0)

    # Steering command.
    # We avoid hard clipping and instead use a smooth limiter to keep
    # delta away from +-pi/2 (tan() blow-up in the bicycle model).
    delta_max = 1.2
    delta = delta_max * np.tanh((k_delta * alpha) / delta_max)

    if dist < 0.05:
        v = 0.0
        delta = 0.0

    return np.array([v, delta], dtype=float)


def controller_unicycle(goal, state):
    """control = [v, omega]"""
    x, y, th = state
    gx, gy, _ = goal

    dx = gx - x
    dy = gy - y
    rho = np.hypot(dx, dy)
    ang_to_goal = np.arctan2(dy, dx)
    alpha = wrap_u(ang_to_goal - th)

    k_rho = 1.2
    k_alpha = 3.0

    v = k_rho * rho
    omega = k_alpha * alpha

    if rho < 0.01:
        v = 0.0
        omega = 0.0

    return np.array([v, omega], dtype=float)


def controller_omni(goal, state):
    """control = [vx, vy, omega] in WORLD frame (matches your omni model)."""
    x, y, th = state
    gx, gy, gth = goal

    ex = gx - x
    ey = gy - y
    eth = wrap_o(gth - th)

    k_p = 1.5
    k_th = 2.0

    vx = k_p * ex
    vy = k_p * ey
    omega = k_th * eth

    if np.hypot(ex, ey) < 0.01 and abs(eth) < 0.01:
        vx = 0.0
        vy = 0.0
        omega = 0.0

    return np.array([vx, vy, omega], dtype=float)


def run():
    sim_iter = int(t_max / Ts) + 1

    state = init_state.copy()
    state_hist = np.zeros((sim_iter, 3))
    goal_hist = np.zeros((sim_iter, 3))

    if MODE in ("unicycle", "ackermann"):
        u_hist = np.zeros((sim_iter, 2))
    else:
        u_hist = np.zeros((sim_iter, 3))

    # Visualization
    if SHOW_2D:
        use_icon = MODE in ("unicycle", "omnidirectional", "ackermann")
        vis = MobileRobotPlotter2D(mode=MODE, use_icon=use_icon)
        vis.set_field(field_x, field_y)
        vis.show_goal(goal_state)

    for it in range(sim_iter):
        t = it * Ts

        state_hist[it] = state
        goal_hist[it] = goal_state

        if MODE == "unicycle":
            u = controller_unicycle(goal_state, state)
            u_hist[it] = u
            state = unicycle_step(state, u, Ts)
        elif MODE == "ackermann":
            u = controller_ackermann(goal_state, state)
            u_hist[it] = u
            state = ackermann_step(state, u, Ts, L=0.3)
        else:
            u = controller_omni(goal_state, state)
            u_hist[it] = u
            state = omni_step(state, u, Ts)

        if SHOW_2D:
            vis.update_time_stamp(t)
            vis.update_goal(goal_state)
            vis.update_trajectory(state_hist[:it+1], control=u)
            plt.pause(1e-3)

    return state_hist, goal_hist, u_hist


if __name__ == "__main__":
    state_hist, goal_hist, u_hist = run()
    t = np.arange(len(state_hist)) * Ts

    # Plot inputs
    plt.figure()
    if MODE == "unicycle":
        plt.plot(t, u_hist[:, 0], label="v [m/s]")
        plt.plot(t, u_hist[:, 1], label="omega [rad/s]")
    elif MODE == "ackermann":
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

    # Plot states
    plt.figure()
    plt.plot(t, state_hist[:, 0], label="px [m]")
    plt.plot(t, state_hist[:, 1], label="py [m]")
    plt.plot(t, state_hist[:, 2], label="theta [rad]")
    plt.plot(t, goal_hist[:, 0], ":", label="goal px")
    plt.plot(t, goal_hist[:, 1], ":", label="goal py")
    plt.plot(t, goal_hist[:, 2], ":", label="goal theta")
    plt.xlabel("t [s]")
    plt.ylabel("state")
    plt.grid(True)
    plt.legend()

    plt.show()
