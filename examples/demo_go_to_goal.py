import os
import sys

# This setting is to allow imports from the parent directory
# Any suggestion to change this?
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from visualization.plotter2d import MobileRobotPlotter2D
from controllers import DetectObstacle, constant_goal, controller_pose_p, plot_standard_results, select_model
from utils.rangesensor import compute_sensor_endpoints


# ----------------------------
# Choose robot type here
# ----------------------------
MODE = "omnidirectional"          # "unicycle", "omnidirectional", or "ackermann"
SHOW_2D = True
SHOW_SENSOR = True

# Simulation settings
Ts = 0.01
t_max = 10.0

# Initial state and goal (state = [px, py, theta])
init_state = np.array([-2.0, -1.5, 0.0])
goal_state_0 = np.array([1.5, 1.0, 0.0])

# Ackermann wheelbase used by the kinematic model
L_ACK = 0.3

# Visualization field limits
field_x = (-2.5, 2.5)
field_y = (-2.0, 2.0)

# Range sensor settings (used only when SHOW_SENSOR=True)
SENSING_RANGE = 1.0
SENSOR_RESOLUTION = np.pi / 8

# Optional: register obstacle boundary vertices (N,2 or N,3). If None, scan shows max-range circle.
OBSTACLE_VERTICES = None


def run():
    sim_iter = int(t_max / Ts) + 1

    get_goal = constant_goal(goal_state_0)
    u_dim, step_fn, wrap_fn = select_model(MODE, L_ack=L_ACK)

    state = init_state.copy()
    state_hist = np.zeros((sim_iter, 3))
    goal_hist = np.zeros((sim_iter, 3))

    u_hist = np.zeros((sim_iter, u_dim))

    detector = None
    if SHOW_SENSOR:
        detector = DetectObstacle(detect_max_dist=SENSING_RANGE, angle_res_rad=SENSOR_RESOLUTION)
        if OBSTACLE_VERTICES is not None:
            detector.register_obstacle_bounded(np.asarray(OBSTACLE_VERTICES, dtype=float))

    # Visualization
    if SHOW_2D:
        vis = MobileRobotPlotter2D(mode=MODE, use_icon=True)
        vis.set_field(field_x, field_y)
        vis.show_goal(get_goal(0.0))

        sensor_pl = None
        if SHOW_SENSOR:
            sensor_pl, = vis.ax.plot([], [], ".", color="k", markersize=4, label="range scan")
            if OBSTACLE_VERTICES is not None:
                v = np.asarray(OBSTACLE_VERTICES, dtype=float)
                vis.ax.plot(v[:, 0], v[:, 1], "--r", label="obstacle")
            vis.ax.legend(loc="upper right")

    for it in range(sim_iter):
        t = it * Ts

        goal_state = get_goal(t)

        state_hist[it] = state
        goal_hist[it] = goal_state

        u = controller_pose_p(MODE, goal_state, state)
        u_hist[it] = u

        if SHOW_2D and SHOW_SENSOR:
            dist = detector.get_sensing_data(state[0], state[1], state[2])
            endpoints = compute_sensor_endpoints(state, dist, sensor_resolution=SENSOR_RESOLUTION)
            sensor_pl.set_data(endpoints[:, 0], endpoints[:, 1])

        state = step_fn(state, u, Ts)
        state[2] = wrap_fn(state[2])

        if SHOW_2D:
            vis.update_time_stamp(t)
            vis.update_goal(goal_state)
            vis.update_trajectory(state_hist[:it+1], control=u)
            plt.pause(1e-3)

    return state_hist, goal_hist, u_hist


if __name__ == "__main__":
    state_hist, goal_hist, u_hist = run()
    plot_standard_results(mode=MODE, Ts=Ts, state_hist=state_hist, goal_hist=goal_hist, u_hist=u_hist)
    plt.show()
