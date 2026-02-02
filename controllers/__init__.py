"""Controllers (control laws) for RoboSimKit demos.
"""

from .go_to_goal import controller_ackermann, controller_omni, controller_unicycle
from .demo_utils import constant_goal, compute_go_to_goal_control, plot_standard_results, select_model
from .follow_figure8 import (
    controller_follow_figure8,
    figure8_goal,
    figure8_reference,
    follow_figure8_step,
)

__all__ = [
    "controller_ackermann",
    "controller_omni",
    "controller_unicycle",
    "constant_goal",
    "compute_go_to_goal_control",
    "plot_standard_results",
    "select_model",
    "controller_follow_figure8",
    "figure8_goal",
    "figure8_reference",
    "follow_figure8_step",
]
