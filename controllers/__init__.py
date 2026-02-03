"""Controllers (control laws) for RoboSimKit demos.
"""

from .go_to_goal import controller_ackermann, controller_omni, controller_pose_p, controller_unicycle
from .sim_utils import constant_goal, compute_go_to_goal_control, plot_standard_results, select_model
from .trajectory_generator import (
    figure8_goal,
    figure8_twist,
    track_reference_step,
)
from .obstacle_detection import DetectObstacle

__all__ = [
    "controller_ackermann",
    "controller_omni",
    "controller_pose_p",
    "controller_unicycle",
    "constant_goal",
    "compute_go_to_goal_control",
    "plot_standard_results",
    "select_model",
    "figure8_goal",
    "figure8_twist",
    "track_reference_step",
    "DetectObstacle",
]
