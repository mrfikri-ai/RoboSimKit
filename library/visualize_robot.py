import numpy as np
import matplotlib.pyplot as plt

# Visualization on 2D plot
class sim_mobile_robot: 

  # Initialization, this only runs one time
  def __init__(self, mode=None):
    # Generate the simulation window and plot initial objects
    self.fig = plt.figure(1)
    self.ax  = plt.gca()
    self.ax.set(xlabel = "x [m]", ylabel = "y [m]")
    self.ax.set_aspect('equal', adjustable ='box', anchor='C')
    plt.tight_layout()

#
