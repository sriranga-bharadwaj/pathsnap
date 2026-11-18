"""Small vectorised geometry helpers (all coordinates in projected metres)."""
import numpy as np


def point_to_segments(px, py, x0, y0, x1, y1):
    """Distance from point(s) (px, py) to segment(s) (x0,y0)-(x1,y1), with numpy broadcasting.

    Returns (dist, frac): frac in [0, 1] is the position of the closest point along the
    segment (0 = start node, 1 = end node). Zero-length segments give frac = 0.
    """
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(L2 > 0, ((px - x0) * dx + (py - y0) * dy) / L2, 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    qx, qy = x0 + frac * dx, y0 + frac * dy
    return np.hypot(px - qx, py - qy), frac


def angle_diff(a, b):
    """Absolute difference between two angles in radians, folded into [0, pi]."""
    d = np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return d
