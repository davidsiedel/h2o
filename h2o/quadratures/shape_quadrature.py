from numpy import ndarray
import h2o.quadratures.gauss.gauss_segment as gauss_segment
import h2o.quadratures.gauss.gauss_triangle as gauss_triangle
from h2o.h2o import *


class ShapeQuadrature:
    quadrature_points: ndarray
    quadrature_weights: ndarray

    def __init__(
        self,
        shape_type: ShapeType,
        shape_vertices: ndarray,
        shape_volume: float,
        integration_order: int,
        quadrature_type: QuadratureType = QuadratureType.GAUSS,
    ):
        """

        Args:
            shape_type:
            shape_vertices:
            shape_volume:
            integration_order:
            quadrature_type:
        """
        if quadrature_type == QuadratureType.GAUSS:
            if shape_type == ShapeType.SEGMENT:
                if not shape_vertices.shape[1] == 2:
                    ValueError("wrong number of vertices")
                (quadrature_reference_points, quadrature_reference_weights) = gauss_segment.get_segment_quadrature_data(
                    integration_order
                )
            elif shape_type == ShapeType.TRIANGLE:
                if not shape_vertices.shape[1] == 3:
                    ValueError("wrong number of vertices")
                (
                    quadrature_reference_points,
                    quadrature_reference_weights,
                ) = gauss_triangle.get_triangle_quadrature_data(integration_order)
            else:
                raise KeyError("unsupported shape")
        else:
            raise KeyError("unsupported quadrature")
        self.quadrature_points = (quadrature_reference_points @ shape_vertices.T).T
        self.quadrature_weights = shape_volume * quadrature_reference_weights

    @staticmethod
    def get_number_of_quadrature_points(
        shape_type: ShapeType, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
    ) -> int:
        """

        Args:
            shape_type:
            integration_order:
            quadrature_type:

        Returns:

        """
        if quadrature_type == QuadratureType.GAUSS:
            if shape_type == ShapeType.SEGMENT:
                number_of_quadrature_points = gauss_segment.get_number_of_quadrature_points_in_segment(
                    integration_order
                )
            elif shape_type == ShapeType.TRIANGLE:
                number_of_quadrature_points = gauss_triangle.get_number_of_quadrature_points_in_triangle(
                    integration_order
                )
            else:
                raise KeyError("unsupported shape")
        else:
            raise KeyError("unsupported quadrature type")
        return number_of_quadrature_points
