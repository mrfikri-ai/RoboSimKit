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

from controllers import controller_ackermann, controller_omni, controller_unicycle

# ----------------------------
# Choose robot type here
# ----------------------------
MODE = "ackermann"          # "unicycle", "omnidirectional", or "ackermann"
SHOW_2D = True

# Simulation settings
Ts = 0.01
t_max = 10.0

# Initial state and goal (state = [px, py, theta])
init_state = np.array([-2.0, -1.5, 0.0])
goal_state = np.array([1.5, 1.0, 0.0])

# Visualization field limits
field_x = (-2.5, 2.5)
field_y = (-2.0, 2.0)


def run():
    sim_iter = int(t_max / Ts) + 1

    if MODE == "unicycle":
        u_dim = 2
        controller = controller_unicycle
        step = unicycle_step
        wrap = wrap_u
    elif MODE == "ackermann":
        u_dim = 2
        controller = controller_ackermann
        step = lambda s, u, ts: ackermann_step(s, u, ts, L=0.3)
        wrap = wrap_a
    elif MODE == "omnidirectional":
        u_dim = 3
        controller = controller_omni
        step = omni_step
        wrap = wrap_o
    else:
        raise ValueError(
            "Invalid MODE. Expected 'unicycle', 'omnidirectional', or 'ackermann'."
        )

    state = init_state.copy()
    state_hist = np.zeros((sim_iter, 3))
    goal_hist = np.zeros((sim_iter, 3))

    u_hist = np.zeros((sim_iter, u_dim))

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

        u = controller(goal_state, state)
        u_hist[it] = u
        state = step(state, u, Ts)
        state[2] = wrap(state[2])

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
