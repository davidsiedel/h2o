from numpy import ndarray

import pythhon.geometry.shapes.point as ipoint
import pythhon.geometry.shapes.segment as isegment
import pythhon.geometry.shapes.triangle as itriangle
from pythhon.parameters import *
from pythhon.quadratures.shapequadrature import ShapeQuadrature


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
        if shape_type == ShapeType.POINT:
            self.centroid = ipoint.get_point_centroid(shape_vertices)
            self.volume = ipoint.get_point_volume()
            self.diameter = ipoint.get_point_diameter()
        elif shape_type == ShapeType.SEGMENT:
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
