import os
import sys

# Ensure repo root is on path so imports work when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from visualization.plotter2d import MobileRobotPlotter2D
from controllers import DetectObstacle, constant_goal, controller_pose_p, plot_standard_results, select_model
from utils.rangesensor import compute_sensor_endpoints


# ----------------------------
# Choose robot type here
# ----------------------------
MODE = "omnidirectional"  # "unicycle", "omnidirectional", or "ackermann"
SHOW_2D = True
SHOW_SENSOR = True

# Simulation settings
Ts = 0.01
t_max = 10.0

# Initial state and goal (state = [px, py, theta])
init_state = np.array([-2.0, -1.5, 0.0], dtype=float)
goal_state_0 = np.array([1.5, 1.0, 0.0], dtype=float)

# Ackermann wheelbase used by the kinematic model
L_ACK = 0.3

# Visualization field limits
field_x = (-2.5, 2.5)
field_y = (-2.0, 2.0)

# ----------------------------
# Obstacle + sensor settings
# ----------------------------
# Obstacle polygon/polyline vertices (world frame). Closed automatically if needed.
obst_vertices = np.array(
    [
        [-1.0, -1.5],
        [1.0, -1.5],
        [1.0, 1.5],
        [-1.0, 1.5],
        [-1.0, 1.0],
        [0.5, 1.0],
        [0.5, -1.0],
        [-1.0, -1.0],
        [-1.0, -1.5],
    ],
    dtype=float,
)

SENSING_RANGE = 1.0  # meters
SENSOR_RESOLUTION = np.pi / 8  # rad

# Obstacle avoidance (reactive) settings
AVOID_DISTANCE = 0.6  # start repelling if obstacle closer than this [m]
AVOID_GAIN = 1.2
TANGENT_GAIN = 0.8
MAX_V_OMNI = 1.0
MAX_OMEGA = 3.0

# For non-omni modes (simple heuristic)
FRONT_ARC_DEG = 30.0
STOP_DISTANCE = 0.5
TURN_GAIN = 1.0


def run():
    sim_iter = int(t_max / Ts) + 1

    get_goal = constant_goal(goal_state_0)
    u_dim, step_fn, wrap_fn = select_model(MODE, L_ack=L_ACK)

    state = init_state.copy()
    state_hist = np.zeros((sim_iter, 3), dtype=float)
    goal_hist = np.zeros((sim_iter, 3), dtype=float)
    u_hist = np.zeros((sim_iter, u_dim), dtype=float)

    # Obstacle detector (360-degree scan)
    detector = DetectObstacle(detect_max_dist=SENSING_RANGE, angle_res_rad=SENSOR_RESOLUTION)
    detector.register_obstacle_bounded(obst_vertices)

    # Visualization
    if SHOW_2D:
        vis = MobileRobotPlotter2D(mode=MODE, use_icon=True)
        vis.set_field(field_x, field_y)
        vis.show_goal(get_goal(0.0))

        # Display obstacle
        vis.ax.plot(obst_vertices[:, 0], obst_vertices[:, 1], "--r", label="obstacle")

        sensor_pl = None
        if SHOW_SENSOR:
            sensor_pl, = vis.ax.plot([], [], ".", color="k", markersize=4, label="range scan")

        vis.ax.legend(loc="upper right")

    for it in range(sim_iter):
        t = it * Ts

        goal_state = get_goal(t)

        state_hist[it] = state
        goal_hist[it] = goal_state

        # Get information from range sensor (360° scan)
        dist = detector.get_sensing_data(state[0], state[1], state[2])

        # Base control (go-to-goal)
        u = controller_pose_p(MODE, goal_state, state)

        # Obstacle avoidance layer
        if MODE == "omnidirectional":
            # Repulsive field in WORLD frame computed from scan rays.
            # Ray direction points away from robot; repulsion points toward -dir.
            n = dist.size
            sensor_angles = np.arange(n, dtype=float) * (2.0 * np.pi / n)
            world_angles = float(state[2]) + sensor_angles
            dirs = np.column_stack([np.cos(world_angles), np.sin(world_angles)])

            # Weight rays that are closer than AVOID_DISTANCE.
            w = np.clip((AVOID_DISTANCE - dist) / max(AVOID_DISTANCE, 1e-9), 0.0, 1.0)
            repulse = (-dirs * w[:, None]).sum(axis=0)

            # Add a tangential component to help slide around obstacles (reduces local minima).
            # Direction (left/right) is chosen based on which side has more clearance.
            if float(np.min(dist)) < AVOID_DISTANCE:
                k = max(1, int(round(np.deg2rad(FRONT_ARC_DEG) / SENSOR_RESOLUTION)))
                left_idx = np.arange(1, k + 1) % n
                right_idx = (-np.arange(1, k + 1)) % n
                left_clear = float(np.min(dist[left_idx]))
                right_clear = float(np.min(dist[right_idx]))

                # Perpendicular to repulse: CCW is "left", CW is "right".
                perp_left = np.array([-repulse[1], repulse[0]], dtype=float)
                perp_right = -perp_left
                perp = perp_left if left_clear >= right_clear else perp_right

                perp_norm = float(np.linalg.norm(perp))
                if perp_norm > 1e-9:
                    tang_strength = TANGENT_GAIN * float(np.clip(w.mean(), 0.0, 1.0))
                    repulse = repulse + tang_strength * (perp / perp_norm)

            u_xy = np.asarray(u[:2], dtype=float) + AVOID_GAIN * repulse
            speed = float(np.linalg.norm(u_xy))
            if speed > MAX_V_OMNI:
                u_xy = u_xy * (MAX_V_OMNI / speed)

            u = np.array([u_xy[0], u_xy[1], float(np.clip(u[2], -MAX_OMEGA, MAX_OMEGA))], dtype=float)

        elif MODE in ("unicycle", "ackermann"):
            # Heuristic: slow down if obstacles are close in front, and turn toward the side with more clearance.
            n = dist.size
            k = max(1, int(round(np.deg2rad(FRONT_ARC_DEG) / SENSOR_RESOLUTION)))

            # Indices around the front beam (index 0 is front).
            left_idx = np.arange(1, k + 1) % n
            right_idx = (-np.arange(1, k + 1)) % n
            front_idx = np.arange(-k, k + 1) % n

            left_min = float(np.min(dist[left_idx]))
            right_min = float(np.min(dist[right_idx]))
            front_min = float(np.min(dist[front_idx]))

            # Reduce forward speed as obstacle gets closer.
            slow = float(np.clip(front_min / max(STOP_DISTANCE, 1e-9), 0.0, 1.0))
            v_goal = float(u[0])
            v = max(0.0, v_goal) * slow

            # Turn away from the closer side.
            eps = 1e-6
            turn = TURN_GAIN * ((1.0 / (right_min + eps)) - (1.0 / (left_min + eps)))

            if MODE == "unicycle":
                omega = float(u[1]) + turn
                omega = float(np.clip(omega, -MAX_OMEGA, MAX_OMEGA))
                u = np.array([v, omega], dtype=float)
            else:
                # Ackermann: steer angle is u[1]
                delta = float(u[1]) + turn
                # Keep steering reasonable (about +/- 45 deg)
                delta = float(np.clip(delta, -np.pi / 4, np.pi / 4))
                u = np.array([v, delta], dtype=float)

        u_hist[it] = u

        # Visualize range scan (points at ray endpoints)
        if SHOW_2D and SHOW_SENSOR:
            endpoints = compute_sensor_endpoints(
                state,
                dist,
                sensor_resolution=SENSOR_RESOLUTION,
            )
            sensor_pl.set_data(endpoints[:, 0], endpoints[:, 1])

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
