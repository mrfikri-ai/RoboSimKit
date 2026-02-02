import os
import sys

# Ensure repo root is on path so imports work when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from visualization.plotter2d import MobileRobotPlotter2D

from controllers import constant_goal, plot_standard_results, select_model


"""Example template for RoboSimKit demos.

Goal
- Provide a consistent skeleton for new algorithms (controllers, planners, etc.)
- Standardize: parameters, control computation, simulation loop, visualization, and plots

How to use
1) Copy this file (recommended):
   - examples/demo_template.py -> examples/demo_my_algorithm.py
2) Choose how you define goal + control:
    - simplest: use get_goal = constant_goal(goal_state_0)
    - implement your algorithm as a controller function (recommended) and import it:
         from controllers.my_controller import compute_control
3) (Optional) replace get_goal with a function for moving goals.
"""


# ----------------------------
# User parameters (edit here)
# ----------------------------
MODE = "unicycle"  # "unicycle", "omnidirectional", or "ackermann"
SHOW_2D = True

Ts = 0.01
t_max = 10.0

# State = [px, py, theta]
init_state = np.array([0.0, 0.0, 0.0], dtype=float)

# Default: constant goal. If you need a moving goal, implement get_goal(t).
goal_state_0 = np.array([1.5, 1.0, 0.0], dtype=float)

# Ackermann wheelbase used by the kinematic model
L_ACK = 0.3

# Visualization field limits
field_x = (-2.5, 2.5)
field_y = (-2.0, 2.0)


# Default goal function (constant). Replace with a function if you want a moving goal.
get_goal = constant_goal(goal_state_0)


# Assign a controller function here (recommended: import from controllers/).
# Signature: compute_control(goal_state, state, t) -> u
compute_control = None


def run():
    """Run the simulation.

    Returns:
      state_hist: (N,3)
      goal_hist:  (N,3)
      u_hist:     (N,u_dim)
    """
    sim_iter = int(t_max / Ts) + 1

    if compute_control is None:
        raise NotImplementedError(
            "Set compute_control to a callable, e.g. 'from controllers.my_controller import compute_control'"
        )

    u_dim, step_fn, wrap_fn = select_model(MODE, L_ack=L_ACK)

    state = init_state.copy()
    state_hist = np.zeros((sim_iter, 3), dtype=float)
    goal_hist = np.zeros((sim_iter, 3), dtype=float)
    u_hist = np.zeros((sim_iter, u_dim), dtype=float)

    # Visualization
    if SHOW_2D:
        vis = MobileRobotPlotter2D(mode=MODE, use_icon=True)
        vis.set_field(field_x, field_y)
        vis.show_goal(get_goal(0.0))

    for it in range(sim_iter):
        t = it * Ts
        goal_state = get_goal(t)

        state_hist[it] = state
        goal_hist[it] = goal_state

        u = np.asarray(compute_control(goal_state, state, t), dtype=float).reshape(u_dim)
        u_hist[it] = u

        state = step_fn(state, u, Ts)
        state[2] = wrap_fn(state[2])

        if SHOW_2D:
            vis.update_time_stamp(t)
            vis.update_goal(goal_state)
            vis.update_trajectory(state_hist[: it + 1], control=u)
            plt.pause(1e-3)

    return state_hist, goal_hist, u_hist


if __name__ == "__main__":
    state_hist, goal_hist, u_hist = run()
    plot_standard_results(mode=MODE, Ts=Ts, state_hist=state_hist, goal_hist=goal_hist, u_hist=u_hist)
    plt.show()
