"""Robot icon visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

class RobotIconArtist:
  def __init__(self, ax, mode: str, scale: float = 2.0):
    self.ax = ax
    self.scale = scale
    self.moro_patch = None

    # Visual style
    self.body_color = "orange"
    self.body_alpha = 1.0
    self.arrow_color = "orange"

    if mode == "omnidirectional":
      self.icon_id = 3
    elif mode == "unicycle":
      self.icon_id = 2
    elif mode == "ackermann":
      self.icon_id = 4
    else:
      raise ValueError("mode is not in the list")

  def update(self, robot_state, control=None):
    px, py, th = robot_state[0], robot_state[1], robot_state[2]

    scale = self.scale
    body_rad = 0.08 * scale
    wheel_size = [0.1 * scale, 0.02 * scale]
    arrow_size = body_rad

    if self.icon_id == 4:
      # Simple car-like icon (Ackermann/bicycle model).
      # Draw a rectangular chassis with four wheels.
      car_length = 0.22 * scale
      car_width = 0.12 * scale
      wheel_len = 0.06 * scale
      wheel_wid = 0.02 * scale

      hx, hy = np.cos(th), np.sin(th)
      lx, ly = -np.sin(th), np.cos(th)

      axle_offset = 0.08 * scale
      track = 0.08 * scale

      # Steering angle (front wheels). If not provided, render straight.
      delta = 0.0
      if control is not None and len(control) >= 2:
        delta = float(control[1])

      # Wheel centers: front/rear x left/right
      wheel_centers = []
      for s_axle in (+1.0, -1.0):
        for s_side in (+1.0, -1.0):
          cx = px + s_axle * axle_offset * hx + s_side * (track / 2.0) * lx
          cy = py + s_axle * axle_offset * hy + s_side * (track / 2.0) * ly
          wheel_centers.append((cx, cy))

      # Wheel angles: rear wheels aligned with body, front wheels steered by delta.
      # Order matches wheel_centers creation:
      #   0 front-left, 1 front-right, 2 rear-left, 3 rear-right
      wheel_angles = [th + delta, th + delta, th, th]

      ar_st = [px, py]
      ar_d = (arrow_size*np.cos(th), arrow_size*np.sin(th))

      if self.moro_patch is None:
        # 0: chassis rectangle, 1: heading arrow, 2..5: wheels
        self.moro_patch = [None] * (2 + len(wheel_centers))

        # Chassis: create axis-aligned then rotate around center
        chassis = plt.Rectangle(
          (px - car_length / 2, py - car_width / 2),
          car_length,
          car_width,
          color=self.body_color,
          alpha=self.body_alpha,
        )
        chassis.set_transform(Affine2D().rotate_around(px, py, th) + self.ax.transData)
        self.moro_patch[0] = self.ax.add_patch(chassis)

        self.moro_patch[1] = plt.quiver(
          ar_st[0], ar_st[1], ar_d[0], ar_d[1],
          scale_units="xy", scale=1, color=self.arrow_color, width=0.1*arrow_size
        )

        for i, ((cx, cy), ang) in enumerate(zip(wheel_centers, wheel_angles)):
          rect = plt.Rectangle(
            (cx - wheel_len / 2, cy - wheel_wid / 2),
            wheel_len,
            wheel_wid,
            color="k",
          )
          rect.set_transform(Affine2D().rotate_around(cx, cy, ang) + self.ax.transData)
          self.moro_patch[2+i] = self.ax.add_patch(rect)
      else:
        chassis = self.moro_patch[0]
        chassis.set_xy((px - car_length / 2, py - car_width / 2))
        chassis.set_transform(Affine2D().rotate_around(px, py, th) + self.ax.transData)

        self.moro_patch[1].set_offsets(ar_st)
        self.moro_patch[1].set_UVC(ar_d[0], ar_d[1])

        for i, ((cx, cy), ang) in enumerate(zip(wheel_centers, wheel_angles)):
          wheel = self.moro_patch[2+i]
          wheel.set_xy((cx - wheel_len / 2, cy - wheel_wid / 2))
          wheel.set_transform(Affine2D().rotate_around(cx, cy, ang) + self.ax.transData)

      return

    if self.icon_id == 2:
      # placing the two wheels on the sides of the body
      wheel_center_angles = [th + np.pi / 2, th - np.pi / 2]
      wheel_angles = [th, th]
    else:
      # Three wheels evenly spaced around the body.
      wheel_center_angles = [(th + i * (2 * np.pi / 3) - np.pi / 2) for i in range(3)]
      # Render wheel rectangles tangential to the body circle.
      wheel_angles = [a + np.pi / 2 for a in wheel_center_angles]

    wheel_centers = [
      (px + body_rad * np.cos(a), py + body_rad * np.sin(a))
      for a in wheel_center_angles
    ]

    ar_st = [px, py]
    ar_d = (arrow_size*np.cos(th), arrow_size*np.sin(th))

    if self.moro_patch is None:
      self.moro_patch = [None] * (2 + len(wheel_angles))
      self.moro_patch[0] = self.ax.add_patch(
        plt.Circle((px, py), body_rad, color=self.body_color, alpha=self.body_alpha)
      )
      self.moro_patch[1] = plt.quiver(ar_st[0], ar_st[1], ar_d[0], ar_d[1],
                  scale_units="xy", scale=1, color=self.arrow_color, width=0.1*arrow_size)
      for i, ((cx, cy), ang) in enumerate(zip(wheel_centers, wheel_angles)):
        rect = plt.Rectangle(
          (cx - wheel_size[0] / 2, cy - wheel_size[1] / 2),
          wheel_size[0],
          wheel_size[1],
          color="k",
        )
        rect.set_transform(Affine2D().rotate_around(cx, cy, ang) + self.ax.transData)
        self.moro_patch[2+i] = self.ax.add_patch(rect)

    else:
      self.moro_patch[0].set(center=(px, py))
      self.moro_patch[1].set_offsets(ar_st)
      self.moro_patch[1].set_UVC(ar_d[0],ar_d[1])
      for i, ((cx, cy), ang) in enumerate(zip(wheel_centers, wheel_angles)):
        wheel = self.moro_patch[2+i]
        wheel.set_xy((cx - wheel_size[0] / 2, cy - wheel_size[1] / 2))
        wheel.set_transform(Affine2D().rotate_around(cx, cy, ang) + self.ax.transData)
