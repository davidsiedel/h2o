from h2o.h2o import *
from h2o.geometry.shape import Shape
from h2o.quadratures.shape_quadrature import ShapeQuadrature
from numpy import ndarray


class Cell:
    shape: Shape
    shape_type: ShapeType
    quadrature_points: ndarray
    quadrature_weights: ndarray

    def __init__(
        self,
        cell_shape_type: ShapeType,
        cell_vertices: ndarray,
        integration_order: int,
        quadrature_type: QuadratureType = QuadratureType.GAUSS,
    ):
        """

        Args:
            cell_shape_type:
            cell_vertices:
            integration_order:
            quadrature_type:
        """
        self.shape = Shape(cell_shape_type, cell_vertices)
        self.shape_type = cell_shape_type
        cell_quadrature = ShapeQuadrature(
            cell_shape_type, cell_vertices, self.shape.volume, integration_order, quadrature_type=quadrature_type,
        )
        self.quadrature_points = cell_quadrature.quadrature_points
        self.quadrature_weights = cell_quadrature.quadrature_weights
