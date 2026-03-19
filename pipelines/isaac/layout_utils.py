from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple


DEFAULT_CLOSED_WALLS: Dict[str, bool] = {
    "behind": True,
    "left": True,
    "right": True,
    "front": False,
}


def normalize_closed_walls(closed_walls: Mapping[str, bool] | None) -> Dict[str, bool]:
    out = dict(DEFAULT_CLOSED_WALLS)
    if not closed_walls:
        return out
    for key in out:
        if key in closed_walls:
            out[key] = bool(closed_walls[key])
    return out


def effective_xy_half_extents(size: Tuple[float, float, float], yaw_rad: float) -> Tuple[float, float]:
    sx, sy = float(size[0]), float(size[1])
    c = abs(math.cos(yaw_rad))
    s = abs(math.sin(yaw_rad))
    return 0.5 * (c * sx + s * sy), 0.5 * (s * sx + c * sy)


def clamp_center_to_room_bounds(
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    yaw_rad: float,
    room_bounds: Tuple[float, float, float, float],
    closed_walls: Mapping[str, bool] | None = None,
) -> Tuple[float, float, float]:
    x, y, z = center
    room_xmin, room_xmax, room_ymin, room_ymax = room_bounds
    walls = normalize_closed_walls(closed_walls)
    hx, hy = effective_xy_half_extents(size, yaw_rad)

    x_min = room_xmin + hx if walls["behind"] else None
    x_max = room_xmax - hx if walls["front"] else None
    y_min = room_ymin + hy if walls["left"] else None
    y_max = room_ymax - hy if walls["right"] else None

    if x_min is not None and x_max is not None and x_min > x_max:
        x = 0.5 * (x_min + x_max)
    else:
        if x_min is not None:
            x = max(x, x_min)
        if x_max is not None:
            x = min(x, x_max)

    if y_min is not None and y_max is not None and y_min > y_max:
        y = 0.5 * (y_min + y_max)
    else:
        if y_min is not None:
            y = max(y, y_min)
        if y_max is not None:
            y = min(y, y_max)

    return (x, y, z)
