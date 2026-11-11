"""Pure alignment guards. No assets or ground truth are read."""
import math
import numpy as np
from .contracts import integer, require


def atd_to_video(frame_id: int) -> int:
    require(integer(frame_id) and frame_id >= 1, "ATD IDs must be positive integers")
    return int(frame_id) - 1


def tagging_to_video(seconds: float) -> int:
    require(not isinstance(seconds, bool) and math.isfinite(seconds) and seconds >= 0, "invalid tagging timestamp")
    return round(seconds * 25)


def validate_frame_sequence(frames, n: int, *, atd: bool = False) -> None:
    require(integer(n) and n > 0, "invalid expected count")
    count = 0
    for count, frame in enumerate(frames, 1):
        require(integer(frame) and frame == count - (0 if atd else 1), "noncontiguous or reordered frame IDs")
    require(count == n, "incorrect frame count")


def project(matrix, x: float, y: float):
    require(np.asarray(matrix).shape == (3, 3) and np.isfinite(matrix).all(), "invalid projection matrix")
    point = matrix @ np.array([x, y, 1.0])
    require(np.isfinite(point).all() and abs(point[2]) > 1e-12, "invalid projection denominator")
    return point[:2] / point[2]
