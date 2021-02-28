from numpy import ndarray

import h2o.geometry.shapes.segment as isegment
import h2o.geometry.shapes.triangle as itriangle
from h2o.h2o import *
from h2o.quadratures.shape_quadrature import ShapeQuadrature


class Shape:
    centroid: ndarray
    volume: float
    diameter: float

    def __init__(self, shape_type: ShapeType, shape_vertices: ndarray):
        """

        Args:
            shape_type:
            shape_vertices:
        """
        if shape_type == ShapeType.SEGMENT:
            self.centroid = isegment.get_segment_centroid(shape_vertices)
            self.volume = isegment.get_segment_volume(shape_vertices)
            self.diameter = isegment.get_segment_diameter(shape_vertices)
        elif shape_type == ShapeType.TRIANGLE:
            self.centroid = itriangle.get_triangle_centroid(shape_vertices)
            self.volume = itriangle.get_triangle_volume(shape_vertices)
            self.diameter = itriangle.get_triangle_diameter(shape_vertices)


def get_shape_quadrature_data(
    shape_type: ShapeType, shape_vertices: ndarray, shape_volume: float, integration_order: int,
) -> (ndarray, ndarray):
    """

    Args:
        shape_type:
        shape_vertices:
        shape_volume:
        integration_order:

    Returns:

    """
    shape_quadrature = ShapeQuadrature(shape_type, shape_vertices, shape_volume, integration_order)
    return shape_quadrature.quadrature_points, shape_quadrature.quadrature_weights
