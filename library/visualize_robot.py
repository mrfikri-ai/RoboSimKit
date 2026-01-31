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
    
    # Plot initial value for trajectory and time stamp
    self.traj_pl, = self.ax.plot(0,0, 'b--')
    self.time_txt = self.ax.text(0.78, 0.01, 't = 0 s', color = 'k', fontsize = 'large'
        horizontalalignment = 'left', verticalalignment='bottom', transform = self.ax.transAxes)
