from __future__ import annotations

import numpy as np


class DetectObstacle:
    """Simple 2D range-sensor obstacle detector using line-segment intersections.

    This is a light port of the classic FunMoRo-style `DetectObstacle` helper.

    - Obstacles are registered as polylines/polygons (vertices in world frame).
    - `get_sensing_data(x, y, theta)` returns a 360° scan (index 0 is forward).

    Notes:
    - Angles are in radians.
    - Distances are capped at `detect_max_dist`.
    """

    def __init__(self, detect_max_dist: float = 10.0, angle_res_rad: float = np.pi / 180):
        if detect_max_dist <= 0:
            raise ValueError("detect_max_dist must be > 0")
        if angle_res_rad <= 0:
            raise ValueError("angle_res_rad must be > 0")

        self.__max_dist = float(detect_max_dist)
        self.__res_rad = float(angle_res_rad)

        self.__sens_num = int(round(2.0 * np.pi / self.__res_rad))
        if self.__sens_num <= 0:
            raise ValueError("angle_res_rad produced 0 sensors")

        self.__sens_linspace = np.linspace(0.0, 2.0 * np.pi, num=self.__sens_num, endpoint=False)

        # Store the obstacle as line segments (x1, y1, x2, y2)
        self.__line_segment_2D = np.zeros((0, 4), dtype=float)
        self.__y1_min_y2: np.ndarray | None = None
        self.__x1_min_x2: np.ndarray | None = None

    @property
    def max_distance(self) -> float:
        return self.__max_dist

    @property
    def angle_resolution(self) -> float:
        return self.__res_rad

    @property
    def sensor_count(self) -> int:
        return self.__sens_num

    def register_obstacle_bounded(self, vertices: np.ndarray) -> None:
        """Register an obstacle boundary described by vertices.

        Args:
            vertices: array of shape (N, 2) or (N, 3). Only first two columns are used.
                      Consecutive pairs define segments. If the boundary is not closed,
                      you should repeat the first point at the end.
        """

        v = np.asarray(vertices, dtype=float)
        if v.ndim != 2 or v.shape[0] < 2:
            raise ValueError("vertices must be an array of shape (N, 2) or (N, 3) with N>=2")

        xy = v[:, :2]
        new_line_segment = np.zeros((xy.shape[0] - 1, 4), dtype=float)
        new_line_segment[:, :2] = xy[:-1]
        new_line_segment[:, 2:] = xy[1:]

        self.__line_segment_2D = np.vstack((self.__line_segment_2D, new_line_segment))
        self.__update_basic_comp()

    def __update_basic_comp(self) -> None:
        self.__y1_min_y2 = self.__line_segment_2D[:, 1] - self.__line_segment_2D[:, 3]
        self.__x1_min_x2 = self.__line_segment_2D[:, 0] - self.__line_segment_2D[:, 2]

    def get_sensing_data(self, posx: float, posy: float, theta_rad: float) -> np.ndarray:
        """Return 360° range scan distances from pose (posx, posy, theta_rad)."""

        if self.__line_segment_2D.shape[0] == 0:
            return np.full(self.__sens_num, self.__max_dist, dtype=float)

        if self.__y1_min_y2 is None or self.__x1_min_x2 is None:
            self.__update_basic_comp()

        # The computation relies on intersection between sensing line-segment and obstacle segments.
        # Follows https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        m = self.__sens_num
        n = self.__line_segment_2D.shape[0]

        n_0 = np.repeat(0.0, n)
        n_1 = np.repeat(1.0, n)

        sensing_angle_rad = float(theta_rad) + self.__sens_linspace
        m_x4_min_x3 = self.__max_dist * np.cos(sensing_angle_rad)
        m_y4_min_y3 = self.__max_dist * np.sin(sensing_angle_rad)

        n_x1_min_x3 = self.__line_segment_2D[:, 0] - np.repeat(float(posx), n)
        n_y1_min_y3 = self.__line_segment_2D[:, 1] - np.repeat(float(posy), n)

        # Loop over each sensing direction
        u_all = np.repeat(1.0, m)
        for i in range(m):
            n_x3_min_x4 = -np.repeat(m_x4_min_x3[i], n)
            n_y3_min_y4 = -np.repeat(m_y4_min_y3[i], n)

            t_upper = (n_x1_min_x3 * n_y3_min_y4) - (n_y1_min_y3 * n_x3_min_x4)
            u_upper = (n_x1_min_x3 * self.__y1_min_y2) - (n_y1_min_y3 * self.__x1_min_x2)
            lower = (self.__x1_min_x2 * n_y3_min_y4) - (self.__y1_min_y2 * n_x3_min_x4)

            with np.errstate(divide="ignore", invalid="ignore"):
                t = t_upper / lower
                u = u_upper / lower

            t_idx = np.logical_and(t >= n_0, t <= n_1)
            u_idx = np.logical_and(u >= n_0, u <= n_1)
            idx = np.logical_and(t_idx, u_idx)
            if np.any(idx):
                u_all[i] = float(np.min(u[idx]))

        return self.__max_dist * u_all


__all__ = ["DetectObstacle"]
