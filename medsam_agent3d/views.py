"""Coordinate transforms for clicks on individual displayed image panels.

This module uses only the Python standard library so coordinate grounding can
be tested without loading image/model dependencies. Coordinates exclude titles,
ticks, panel borders, and padding: (0,0) is the top-left image pixel and
(999,999) the bottom-right image pixel.
"""
import math
import operator
from numbers import Real


def view_point_to_voxel(view, point_2d, center, shape):
    """Map normalized panel (u,v) to a ZYX voxel on the displayed slice.

    axial: z=center.z, y=v, x=u
    coronal: z=v, y=center.y, x=u
    sagittal: z=v, y=u, x=center.x
    """
    if view not in ('axial', 'coronal', 'sagittal'):
        raise ValueError('view must be axial, coronal, or sagittal')
    if not isinstance(point_2d, (list, tuple)) or len(point_2d) != 2:
        raise ValueError('point_2d must contain normalized (u,v) coordinates')
    if any(isinstance(v, bool) or not isinstance(v, Real) or not 0 <= v <= 999 or not math.isfinite(v) or int(v) != v for v in point_2d):
        raise ValueError('point_2d coordinates must be integers in [0,999]')
    if len(shape) != 3 or len(center) != 3:
        raise ValueError('shape and center must each contain three ZYX values')
    try:
        if any(isinstance(v, bool) for v in (*shape, *center)):
            raise TypeError
        shape = tuple(operator.index(v) for v in shape)
        center = tuple(operator.index(v) for v in center)
    except TypeError as exc:
        raise ValueError('shape and center must contain integers') from exc
    if any(size <= 0 for size in shape):
        raise ValueError('shape dimensions must be positive')
    if any(not 0 <= position < size for position, size in zip(center, shape)):
        raise ValueError('center must lie inside the volume')
    u, v = point_2d
    def index(value, size):
        return round(value / 999. * (size - 1))
    if view == 'axial':
        return center[0], index(v, shape[1]), index(u, shape[2])
    if view == 'coronal':
        return index(v, shape[0]), center[1], index(u, shape[2])
    return index(v, shape[0]), index(u, shape[1]), center[2]
