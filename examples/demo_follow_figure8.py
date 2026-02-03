import os
import sys

# Ensure repo root is on path so imports work when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from visualization.plotter2d import MobileRobotPlotter2D

from controllers import DetectObstacle, track_reference_step, figure8_goal, figure8_twist, plot_standard_results, select_model
from utils.rangesensor import compute_sensor_endpoints


# ----------------------------
# Choose robot type here
# ----------------------------
MODE = "omnidirectional"  # "unicycle", "omnidirectional", or "ackermann"
SHOW_2D = True
SHOW_SENSOR = True

# Simulation settings
Ts = 0.01
t_max = 20.0

# Initial state (state = [px, py, theta])
init_state = np.array([0.0, 0.0, 0.0], dtype=float)

# Trajectory settings (figure-8 / lemniscate-like)
A = 1.5  # x amplitude [m]
B = 1.0  # y amplitude [m]
w = 0.6  # rad/s

# Ackermann wheelbase
L_ACK = 0.3

# Feedback gains (tune once; reused for all models)
K_POS = 1.2
K_THETA = 3.0

# Optional feedforward (helps time-varying reference tracking, especially for omni)
USE_FEEDFORWARD = True

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

    u_dim, step_fn, wrap_fn = select_model(MODE, L_ack=L_ACK)

    state = init_state.copy()
    state_hist = np.zeros((sim_iter, 3), dtype=float)
    ref_hist = np.zeros((sim_iter, 3), dtype=float)
    u_hist = np.zeros((sim_iter, u_dim), dtype=float)

    detector = None
    if SHOW_SENSOR:
        detector = DetectObstacle(detect_max_dist=SENSING_RANGE, angle_res_rad=SENSOR_RESOLUTION)
        if OBSTACLE_VERTICES is not None:
            detector.register_obstacle_bounded(np.asarray(OBSTACLE_VERTICES, dtype=float))

    # Visualization
    if SHOW_2D:
        vis = MobileRobotPlotter2D(mode=MODE, use_icon=True)
        vis.set_field(field_x, field_y)

        # Draw the full reference path (figure-8) so tracking is easy to see.
        t_grid = np.arange(sim_iter) * Ts
        ref_xy = np.array([figure8_goal(ti, A=A, B=B, w=w)[:2] for ti in t_grid], dtype=float)
        vis.ax.plot(ref_xy[:, 0], ref_xy[:, 1], ":", color="r", label="reference")

        sensor_pl = None
        if SHOW_SENSOR:
            sensor_pl, = vis.ax.plot([], [], ".", color="k", markersize=4, label="range scan")
            if OBSTACLE_VERTICES is not None:
                v = np.asarray(OBSTACLE_VERTICES, dtype=float)
                vis.ax.plot(v[:, 0], v[:, 1], "--r", label="obstacle")

        vis.ax.legend(loc="upper right")

    for it in range(sim_iter):
        t = it * Ts

        desired_state, u = track_reference_step(
            MODE,
            state,
            t,
            get_reference=(lambda ti: figure8_goal(ti, A=A, B=B, w=w)),
            K_POS=K_POS,
            K_THETA=K_THETA,
            L_ack=L_ACK,
        )

        if USE_FEEDFORWARD and MODE == "omnidirectional":
            # Add feedforward twist for better trajectory tracking.
            u = np.asarray(u, dtype=float).copy()
            u = u + figure8_twist(t, A=A, B=B, w=w)

        state_hist[it] = state
        ref_hist[it] = desired_state

        u_hist[it] = u

        if SHOW_2D and SHOW_SENSOR:
            dist = detector.get_sensing_data(state[0], state[1], state[2])
            endpoints = compute_sensor_endpoints(state, dist, sensor_resolution=SENSOR_RESOLUTION)
            sensor_pl.set_data(endpoints[:, 0], endpoints[:, 1])

        state = step_fn(state, u, Ts)
        state[2] = wrap_fn(state[2])

        if SHOW_2D:
            vis.update_time_stamp(t)
            vis.update_goal(desired_state)
            vis.update_trajectory(state_hist[: it + 1], control=u)
            plt.pause(1e-3)

    return state_hist, ref_hist, u_hist


if __name__ == "__main__":
    state_hist, ref_hist, u_hist = run()
    plot_standard_results(
        mode=MODE,
        Ts=Ts,
        state_hist=state_hist,
        goal_hist=ref_hist,
        u_hist=u_hist,
        goal_label="ref",
    )
    plt.show()
