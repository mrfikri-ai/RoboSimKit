import os
import sys

# Ensure repo root is on path so imports work when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from visualization.plotter2d import MobileRobotPlotter2D

from controllers import controller_ackermann, controller_omni, controller_unicycle

from models.unicycle import step as unicycle_step, wrap_angle as wrap_u
from models.omni import step as omni_step, wrap_angle as wrap_o
from models.ackermann import step as ackermann_step, wrap_angle as wrap_a


# ----------------------------
# Choose robot type here
# ----------------------------
MODE = "unicycle"  # "unicycle", "omnidirectional", or "ackermann"
SHOW_2D = True

# Simulation settings
Ts = 0.01
t_max = 20.0

# Trajectory settings (figure-8 / lemniscate-like)
A = 1.5  # x amplitude [m]
B = 1.0  # y amplitude [m]
w = 0.6  # rad/s

# Ackermann wheelbase
L_ACK = 0.3

# Feedback gains (tune once; reused for all models)
K_POS = 1.2
K_THETA = 3.0

# Visualization field limits
field_x = (-2.5, 2.5)
field_y = (-2.0, 2.0)


def figure8_reference(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (desired_state, feedforward_u).

    desired_state: [x_d, y_d, theta_d]
    feedforward_u:
      - omni:      [vx_d, vy_d, omega_d]
      - unicycle:  [v_d, omega_d]
      - ackermann: [v_d, delta_d]

    The trajectory is:
      x = A sin(w t)
      y = (B/2) sin(2 w t)
    """
    x = A * np.sin(w * t)
    y = 0.5 * B * np.sin(2.0 * w * t)

    x_d = A * w * np.cos(w * t)
    y_d = B * w * np.cos(2.0 * w * t)

    x_dd = -A * (w**2) * np.sin(w * t)
    y_dd = -2.0 * B * (w**2) * np.sin(2.0 * w * t)

    theta_d = np.arctan2(y_d, x_d)

    v = float(np.hypot(x_d, y_d))

    # Curvature of a planar parametric curve.
    # kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)
    denom = (x_d * x_d + y_d * y_d) ** 1.5
    if denom == 0.0:
        kappa = 0.0
    else:
        kappa = float((x_d * y_dd - y_d * x_dd) / denom)

    omega = v * kappa

    desired_state = np.array([x, y, theta_d], dtype=float)

    if MODE == "unicycle":
        u_ff = np.array([v, omega], dtype=float)
    elif MODE == "ackermann":
        delta = float(np.arctan(L_ACK * kappa))
        u_ff = np.array([v, delta], dtype=float)
    else:
        u_ff = np.array([x_d, y_d, omega], dtype=float)

    return desired_state, u_ff


def run():
    sim_iter = int(t_max / Ts) + 1

    state = np.array([0.0, 0.0, 0.0], dtype=float)

    state_hist = np.zeros((sim_iter, 3))
    ref_hist = np.zeros((sim_iter, 3))

    if MODE in ("unicycle", "ackermann"):
        u_hist = np.zeros((sim_iter, 2))
    else:
        u_hist = np.zeros((sim_iter, 3))

    # Visualization
    if SHOW_2D:
        use_icon = MODE in ("unicycle", "omnidirectional", "ackermann")
        vis = MobileRobotPlotter2D(mode=MODE, use_icon=use_icon)
        vis.set_field(field_x, field_y)

        # Draw the full reference path (figure-8) so tracking is easy to see.
        t_grid = np.arange(sim_iter) * Ts
        ref_xy = np.array([figure8_reference(ti)[0][:2] for ti in t_grid], dtype=float)
        vis.ax.plot(ref_xy[:, 0], ref_xy[:, 1], ":", color="r", label="reference")
        vis.ax.legend(loc="upper right")

    for it in range(sim_iter):
        t = it * Ts

        desired_state, u_ff = figure8_reference(t)

        state_hist[it] = state
        ref_hist[it] = desired_state

        if MODE == "unicycle":
            u_fb = controller_unicycle(desired_state, state, k_rho=K_POS, k_alpha=K_THETA)
            u = u_ff + u_fb
            u_hist[it] = u
            state = unicycle_step(state, u, Ts)
            state[2] = wrap_u(state[2])
        elif MODE == "ackermann":
            u_fb = controller_ackermann(desired_state, state, k_v=K_POS, k_delta=K_THETA)
            u = u_ff + u_fb
            u_hist[it] = u
            state = ackermann_step(state, u, Ts, L=L_ACK)
            state[2] = wrap_a(state[2])
        else:
            u_fb = controller_omni(desired_state, state, k_p=K_POS, k_th=K_THETA)
            u = u_ff + u_fb
            u_hist[it] = u
            state = omni_step(state, u, Ts)
            state[2] = wrap_o(state[2])

        if SHOW_2D:
            vis.update_time_stamp(t)
            vis.update_trajectory(state_hist[: it + 1], control=u)
            plt.pause(1e-3)

    return state_hist, ref_hist, u_hist


if __name__ == "__main__":
    state_hist, ref_hist, u_hist = run()
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

    # Plot states vs reference
    plt.figure()
    plt.plot(t, state_hist[:, 0], label="px [m]")
    plt.plot(t, state_hist[:, 1], label="py [m]")
    plt.plot(t, state_hist[:, 2], label="theta [rad]")
    plt.plot(t, ref_hist[:, 0], ":", label="ref px")
    plt.plot(t, ref_hist[:, 1], ":", label="ref py")
    plt.plot(t, ref_hist[:, 2], ":", label="ref theta")
    plt.xlabel("t [s]")
    plt.ylabel("state")
    plt.grid(True)
    plt.legend()

    plt.show()
