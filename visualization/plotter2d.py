"""2D visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt

from .robot_icon import RobotIconArtist


class MobileRobotPlotter2D:
      def __init__(self, mode=None, use_icon=True):
            self.fig = plt.figure(1)
            self.ax = plt.gca()
            self.ax.set(xlabel="x [m]", ylabel="y [m]")
            self.ax.set_aspect("equal", adjustable="box", anchor="C")
            plt.tight_layout()

            self.traj_pl, = self.ax.plot([], [], "b--")
            self.time_txt = self.ax.text(
                  0.78,
                  0.01,
                  "t = 0 s",
                  color="k",
                  fontsize="large",
                  horizontalalignment="left",
                  verticalalignment="bottom",
                  transform=self.ax.transAxes,
            )

            self.goal_pl = None
            self.icon_artist = None
            self.pos_pl = None

            if use_icon and mode in ("unicycle", "omnidirectional", "ackermann"):
                  self.icon_artist = RobotIconArtist(self.ax, mode=mode)
                  self.icon_artist.update(np.zeros(3))
            else:
                  self.pos_pl, = self.ax.plot(0, 0, "b", marker="X", markersize=10)

      def set_field(self, xlim, ylim):
            self.ax.set(xlim=xlim, ylim=ylim)

      def show_goal(self, goal_state):
            self.update_goal(goal_state)

      def update_goal(self, goal_state):
            """Create or update the goal marker."""
            arrow_size = 0.2
            dx = arrow_size * np.cos(goal_state[2])
            dy = arrow_size * np.sin(goal_state[2])

            if self.goal_pl is None:
                  self.goal_pl = self.ax.quiver(
                        goal_state[0],
                        goal_state[1],
                        dx,
                        dy,
                        scale_units="xy",
                        scale=1,
                        color="r",
                        width=0.1 * arrow_size,
                  )
            else:
                  self.goal_pl.set_offsets([goal_state[0], goal_state[1]])
                  self.goal_pl.set_UVC(dx, dy)

      def update_time_stamp(self, t):
            self.time_txt.set_text(f"t = {t:.1f} s")

      def update_trajectory(self, state_hist, control=None):
            px = state_hist[:, 0]
            py = state_hist[:, 1]
            state = state_hist[-1]

            self.traj_pl.set_data(px, py)
            if self.icon_artist is not None:
                  self.icon_artist.update(state, control=control)
            else:
                  self.pos_pl.set_data([state[0]], [state[1]])

            plt.pause(1e-6)

