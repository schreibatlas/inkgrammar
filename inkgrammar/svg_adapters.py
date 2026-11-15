from __future__ import annotations

from planarcurve.svg import transform_path as _transform_path
from planarcurve.transforms import Affine2D

from .models import Transform


def to_planar_affine(transform: Transform) -> Affine2D:
    return Affine2D(
        m00=transform.scale_x,
        m01=transform.shear_x * transform.scale_y,
        m10=0.0,
        m11=transform.scale_y,
        dx=transform.dx,
        dy=transform.dy,
    )


def transform_svg_path(path: str, transform: Transform) -> str:
    return _transform_path(path, to_planar_affine(transform))
