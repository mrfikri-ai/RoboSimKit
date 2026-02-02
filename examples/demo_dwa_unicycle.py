import os
import sys

# Ensure repo root is on path so imports work when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import numpy as np
import matplotlib.pyplot as plt

from visualization.plotter2d import MobileRobotPlotter2D
from controllers import controller_unicycle
from controllers import plot_standard_results
from models.unicycle import step as unicycle_step
from utils.angles import wrap_angle

"""Dynamic Window Approach (DWA) demo (standardized example structure).

This demo is intentionally self-contained so other algorithms can copy the structure.

Run:
    python examples/demo_dwa_unicycle.py
"""


# ----------------------------
# User parameters (edit here)
# ----------------------------
Ts = 0.1  # [s] controller + simulation step
SHOW_2D = True
MAX_STEPS = 1000

# Goal stopping / near-goal behavior.
# Note: with discrete time (DT) and finite velocity sampling (cfg['v_reso']), the robot may
# never enter an extremely tiny goal threshold unless we explicitly slow down near the goal.
GOAL_TOL = 0.1
SLOW_RADIUS = 0.5
STOP_ROT_RADIUS = 0.05
GOAL_SWITCH_RADIUS = 0.8

# Initial state of the robot [x, y, yaw, v, w]
init_x5 = np.array([0.0, 0.0, math.pi / 2.0, 0.0, 0.0], dtype=float)

# Goal position and goal state (used by go-to-goal fallback)
goal_xy = np.array([8.0, 6.0], dtype=float)
goal_state_0 = np.array([goal_xy[0], goal_xy[1], 0.0], dtype=float)

# Obstacle positions list [x, y]
obstacles_xy = np.array(
    [
        [0, 2],
        [4, 2],
        [4, 4],
        [5, 4],
        [5, 5],
        [5, 6],
        [5, 9],
        [8, 8],
        [8, 9],
        [7, 9],
        [6, 5],
        [6, 3],
        [6, 8],
        [6, 6],
        [7, 4],
    ],
    dtype=float,
)

# Obstacle radius for collision detection
obstacle_r = 0.6

# Kinematic constraints.
# Note: the original script uses very fine sampling (0.01 m/s and 1 deg).
# For an interactive demo, we default to slightly coarser sampling so it runs fast.
cfg = {
    "max_v": 1.0,
    "max_w": math.radians(20.0),
    "max_acc_v": 0.2,
    "max_acc_w": math.radians(50.0),
    "v_reso": 0.05,
    "w_reso": math.radians(5.0),
}

# Weights + prediction horizon
eval_w = {
    "heading": 0.05,
    "dist": 0.2,
    "velocity": 0.1,
    "to_goal": 0.6,
    "predict_time": 3.0,
}

# Visualization limits
field_x = (-1.0, 11.0)
field_y = (-1.0, 11.0)


def motion_model(x5: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Unicycle motion model with velocity states.

    x5 = [x, y, theta, v, w]
    u  = [v_cmd, w_cmd]

    We treat (v, w) as the applied command for the next step.
    """
    x5 = np.asarray(x5, dtype=float).reshape(5)
    u = np.asarray(u, dtype=float).reshape(2)

    pose = x5[:3]
    pose = unicycle_step(pose, u, dt)
    pose[2] = wrap_angle(pose[2])

    return np.array([pose[0], pose[1], pose[2], u[0], u[1]], dtype=float)


def calc_dynamic_window(x5: np.ndarray, cfg: dict, dt: float) -> np.ndarray:
    """Dynamic window [vmin, vmax, wmin, wmax]."""
    v = float(x5[3])
    w = float(x5[4])

    # Static limits
    vs = np.array([0.0, cfg["max_v"], -cfg["max_w"], cfg["max_w"]], dtype=float)

    # Dynamic limits from acceleration
    vd = np.array(
        [
            v - cfg["max_acc_v"] * dt,
            v + cfg["max_acc_v"] * dt,
            w - cfg["max_acc_w"] * dt,
            w + cfg["max_acc_w"] * dt,
        ],
        dtype=float,
    )

    dw = np.array(
        [max(vs[0], vd[0]), min(vs[1], vd[1]), max(vs[2], vd[2]), min(vs[3], vd[3])],
        dtype=float,
    )
    return dw


def generate_trajectory(
    x5: np.ndarray, v: float, w: float, predict_time: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Forward simulate for predict_time, returning (x5_end, traj_pose[N,3])."""
    time = 0.0
    x = np.asarray(x5, dtype=float).reshape(5)
    traj = [x[:3].copy()]

    u = np.array([v, w], dtype=float)
    while time <= predict_time:
        x = motion_model(x, u, dt)
        traj.append(x[:3].copy())
        time += dt

    return x, np.asarray(traj, dtype=float)


def heading_score(x5_end: np.ndarray, goal_xy: np.ndarray) -> float:
    """Higher is better: heading alignment to goal direction."""
    x, y, theta = x5_end[:3]
    goal_theta = math.atan2(float(goal_xy[1] - y), float(goal_xy[0] - x))
    err = wrap_angle(goal_theta - float(theta))
    # In [0, pi], larger means better aligned.
    return math.pi - abs(err)


def obstacle_distance_score(x5_end: np.ndarray, obstacles_xy: np.ndarray, r: float) -> float:
    """Return minimum clearance to obstacles (capped), higher is better."""
    pos = x5_end[:2].reshape(2, 1)
    if obstacles_xy.size == 0:
        return 2.0 * r

    dists = np.linalg.norm(obstacles_xy.T - pos, axis=0) - r
    dist = float(np.min(dists))

    # Cap to avoid overweighting obstacle-free trajectories.
    return min(dist, 2.0 * r)


def braking_distance(v: float, cfg: dict, dt: float) -> float:
    stop_dist = 0.0
    vel = abs(float(v))
    while vel > 0.0:
        stop_dist += vel * dt
        vel -= cfg["max_acc_v"] * dt
    return stop_dist


def dwa_control(
    x5: np.ndarray,
    cfg: dict,
    goal_xy: np.ndarray,
    eval_w: dict,
    obstacles_xy: np.ndarray,
    obstacle_r: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic Window Approach.

    Returns:
      u = [v, w]
      best_traj_pose[N,3]
    """
    dw = calc_dynamic_window(x5, cfg, dt)

    candidates = []
    trajs = []

    v_vals = np.arange(dw[0], dw[1] + cfg["v_reso"], cfg["v_reso"], dtype=float)
    w_vals = np.arange(dw[2], dw[3] + cfg["w_reso"], cfg["w_reso"], dtype=float)

    for v in v_vals:
        for w in w_vals:
            x_end, traj = generate_trajectory(x5, float(v), float(w), eval_w["predict_time"], dt)

            h = heading_score(x_end, goal_xy)
            dist = obstacle_distance_score(x_end, obstacles_xy, obstacle_r)
            vel = abs(float(v))
            goal_dist = float(np.linalg.norm(x_end[:2] - goal_xy))
            to_goal = 1.0 / (goal_dist + 1e-9)

            stop = braking_distance(vel, cfg, dt)
            if dist <= stop:
                continue

            candidates.append([float(v), float(w), h, dist, vel, to_goal])
            trajs.append(traj)

    if not candidates:
        return np.array([0.0, 0.0], dtype=float), np.asarray([x5[:3].copy()], dtype=float)

    cand = np.asarray(candidates, dtype=float)

    # Normalize each score column (like the original code).
    for col in (2, 3, 4, 5):
        s = float(np.sum(cand[:, col]))
        if s > 0.0:
            cand[:, col] = cand[:, col] / s

    total = (
        eval_w["heading"] * cand[:, 2]
        + eval_w["dist"] * cand[:, 3]
        + eval_w["velocity"] * cand[:, 4]
        + eval_w["to_goal"] * cand[:, 5]
    )

    best_i = int(np.argmax(total))
    u = cand[best_i, 0:2]
    return u.astype(float), trajs[best_i]


def compute_control(x5: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute (u, best_traj_pose) for DWA with a near-goal fallback controller."""
    dist_to_goal = float(np.linalg.norm(x5[:2] - goal_xy))
    goal_state = goal_state_0

    if dist_to_goal < GOAL_SWITCH_RADIUS:
        u = controller_unicycle(goal_state, x5[:3])
        best_traj = np.asarray([x5[:3].copy()], dtype=float)
    else:
        u, best_traj = dwa_control(
            x5,
            cfg,
            goal_xy,
            eval_w,
            obstacles_xy,
            obstacle_r,
            Ts,
        )

    if dist_to_goal < SLOW_RADIUS:
        u = u.copy()
        u[0] = min(float(u[0]), dist_to_goal / Ts)
        if dist_to_goal < STOP_ROT_RADIUS:
            u[1] = 0.0

    return u, best_traj


def run():
    print("Dynamic Window Approach demo start")

    x = init_x5.copy()
    goal_state = goal_state_0

    state_hist = [x[:3].copy()]
    goal_hist = [goal_state.copy()]
    u_hist = [np.zeros(2, dtype=float)]

    if SHOW_2D:
        vis = MobileRobotPlotter2D(mode="unicycle", use_icon=True)
        vis.set_field(field_x, field_y)
        vis.show_goal(goal_state)

        vis.ax.plot(obstacles_xy[:, 0], obstacles_xy[:, 1], "*k", label="obstacles")
        pred_line, = vis.ax.plot([], [], "g--", linewidth=1.0, label="best prediction")
        vis.ax.legend(loc="upper left")

    for step_i in range(MAX_STEPS):
        t = step_i * Ts
        goal_state = goal_state_0
        u, best_traj = compute_control(x, t)

        u_hist.append(u.copy())

        x = motion_model(x, u, Ts)
        state_hist.append(x[:3].copy())
        goal_hist.append(goal_state.copy())

        # goal reached
        if np.linalg.norm(x[:2] - goal_xy) < GOAL_TOL:
            print("Arrive Goal!!")
            break

        if SHOW_2D:
            h = np.asarray(state_hist, dtype=float)
            vis.update_time_stamp(t)
            vis.update_goal(goal_state)
            vis.update_trajectory(h, control=u)
            pred_line.set_data(best_traj[:, 0], best_traj[:, 1])
            plt.pause(1e-3)

    return (
        np.asarray(state_hist, dtype=float),
        np.asarray(goal_hist, dtype=float),
        np.asarray(u_hist, dtype=float),
    )


if __name__ == "__main__":
    state_hist, goal_hist, u_hist = run()

    plot_standard_results(
        mode="unicycle",
        Ts=Ts,
        state_hist=state_hist,
        goal_hist=goal_hist,
        u_hist=u_hist,
        goal_label="goal",
    )
    plt.show()
