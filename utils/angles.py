"""Angle utilities shared across models and controllers."""

from __future__ import annotations

import numpy as np


def wrap_angle(theta: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


__all__ = ["wrap_angle"]
