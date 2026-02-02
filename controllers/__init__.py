"""Controllers (control laws) for RoboSimKit demos.

This package intentionally keeps controllers separate from kinematic models in `models/`.
"""

from .go_to_goal import controller_ackermann, controller_omni, controller_unicycle

__all__ = [
    "controller_ackermann",
    "controller_omni",
    "controller_unicycle",
]
