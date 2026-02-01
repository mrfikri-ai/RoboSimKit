# library/viz/robot_icon.py
import numpy as np
import matplotlib.pyplot as plt

class RobotIconArtist:
  def __init__(self, ax, mode: str, scale: float = 2.0):
    self.ax = ax
    self.scale = scale
    self.moro_pacth = None

    if mode == "omnidirectional":
      self.icon_id = 3
    elif mode == "unicycle":
      self.icon_id = 2
    else:
      raise ValueError("mode is not in the list")

  def update(self, robot_state):
    px, py, th = robot_state[0], robot_state[1], robot_state[2]

    scale = self.scale
    body_rad = 0.08 * scale
    wheel_size = [0.1 * scale, 0.02 * scale]
    arrow_size = body_rad

    if self.icon_id == 2:
      thWh = [th+0, th+np.pi]
    else:
      thWh = [(th + i*(2*np.pi/3) - np.pi/2) for i in range(3)]

    thWh_deg = [np.rad2deg(i) for i in thWh]
    wh_x = [px - body_rad*np.sin(i) - (wheel_size[0]/2)*np.cos(i) + (wheel_size[1]/2)*np.sin(i) for i in thWh]
    wh_y = [py - body_rad*np.cos(i) - (wheel_size[0]/2)*np.sin(i) + (wheel_size[1]/2)*np.cos(i) for i in thWh]

    ar_st = [px, py]
    ar_d = (arrow_size*np.cos(th), arrow_size*np.sin(th))

    if self.moro_patch is None:
      self.moro_patch = [None]*(2+len(thWh))
      self.moro_patch[0] = self.ax.add_patch(plt.Circle((px, py), body_rad, color="#AAAAAAAA"))
      self.moro_patch[1] = plt.quiver(ar_st[0], ar_st[1], ar_d[0], ar_d[1],
                  scale_unit="xy", scale=1, color="b", width=0.1*arrow_size)
      for i in range(len(thWh)):
        self.moro_patch[2+i] = self.ax.add_patch(
          plt.Rectangle((wh_x[i], wh_y[i]), wheel_size[0], wheel_size[1], angle=thWh_deg[i], color="k"))

    else:
      self.moro_patch[0].set(center=(px, py))
      self.moro_patch[1].set_offset(ar_st)
      self.moro_patch[1].set_UVC(ar_d[0],ar_d[1])
      for i in range(len(thWh)):
        self.moro_patch[2+i].set(xy=(wh_x[i], wh_y[i]))
        self.moro_patch[2+i].angle= thWh_deg[i]
