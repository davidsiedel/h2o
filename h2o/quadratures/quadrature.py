import h2o.quadratures.gauss.gauss_segment as gauss_segment
import h2o.quadratures.gauss.gauss_triangle as gauss_triangle
from h2o.quadratures.quadrature_utils import *


def get_quadrature_item(
    shape_type: ShapeType,
    integration_order: int,
    quadrature_item: QuadratureItem,
    quadrature_type: QuadratureType = QuadratureType.GAUSS,
) -> Union[ndarray, int]:
    """

    Args:
        shape_type:
        integration_order:
        quadrature_item:
        quadrature_type:

    Returns:

    """
    if quadrature_type == QuadratureType.GAUSS:
        if shape_type == ShapeType.SEGMENT:
            if quadrature_item in [QuadratureItem.POINTS, QuadratureItem.WEIGHTS]:
                return gauss_segment.get_segment_quadrature(integration_order, quadrature_item)
            elif quadrature_item == QuadratureItem.SIZE:
                return gauss_segment.get_number_of_quadrature_points_in_segment(integration_order)
        elif shape_type == ShapeType.TRIANGLE:
            if quadrature_item in [QuadratureItem.POINTS, QuadratureItem.WEIGHTS]:
                return gauss_triangle.get_triangle_quadrature(integration_order, quadrature_item)
            elif quadrature_item == QuadratureItem.SIZE:
                return gauss_triangle.get_number_of_quadrature_points_in_triangle(integration_order)
        else:
            raise KeyError("unsupported shape")
    else:
        raise KeyError("unsupported quadrature type")


def get_shape_quadrature_points(
    shape_type: ShapeType,
    shape_vertices: ndarray,
    integration_order: int,
    quadrature_type: QuadratureType = QuadratureType.GAUSS,
) -> ndarray:
    """

    Args:
        shape_type:
        shape_vertices:
        integration_order:
        quadrature_type:

    Returns:

    """
    quadrature_reference_points = get_quadrature_item(
        shape_type, integration_order, QuadratureItem.POINTS, quadrature_type=quadrature_type
    )
    quadrature_points = (quadrature_reference_points @ shape_vertices.T).T
    return quadrature_points


def get_shape_quadrature_weights(
    shape_type: ShapeType,
    shape_volume: float,
    integration_order: int,
    quadrature_type: QuadratureType = QuadratureType.GAUSS,
) -> ndarray:
    """

    Args:
        shape_type:
        shape_volume:
        integration_order:
        quadrature_type:

    Returns:

    """
    quadrature_reference_weights = get_quadrature_item(
        shape_type, integration_order, QuadratureItem.WEIGHTS, quadrature_type=quadrature_type
    )
    quadrature_weights = shape_volume * quadrature_reference_weights
    return quadrature_weights


def get_shape_quadrature_size(
    shape_type: ShapeType, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
) -> int:
    """

    Args:
        shape_type:
        integration_order:
        quadrature_type:

    Returns:

    """
    quadrature_size = get_quadrature_item(
        shape_type, integration_order, QuadratureItem.SIZE, quadrature_type=quadrature_type
    )
    return quadrature_size


# class ShapeQuadrature:
#     quadrature_points: ndarray
#     quadrature_weights: ndarray
#
#     def __init__(
#         self,
#         shape_type: ShapeType,
#         shape_vertices: ndarray,
#         shape_volume: float,
#         integration_order: int,
#         quadrature_type: QuadratureType = QuadratureType.GAUSS,
#     ):
#         """
#
#         Args:
#             shape_type:
#             shape_vertices:
#             shape_volume:
#             integration_order:
#             quadrature_type:
#         """
#         if quadrature_type == QuadratureType.GAUSS:
#             if shape_type == ShapeType.SEGMENT:
#                 if not shape_vertices.shape[1] == 2:
#                     ValueError("wrong number of vertices")
#                 (quadrature_reference_points, quadrature_reference_weights) = gauss_segment.get_segment_quadrature(
#                     integration_order
#                 )
#             elif shape_type == ShapeType.TRIANGLE:
#                 if not shape_vertices.shape[1] == 3:
#                     ValueError("wrong number of vertices")
#                 (quadrature_reference_points, quadrature_reference_weights) = gauss_triangle.get_triangle_quadrature(
#                     integration_order
#                 )
#             else:
#                 raise KeyError("unsupported shape")
#         else:
#             raise KeyError("unsupported quadrature")
#         self.quadrature_points = (quadrature_reference_points @ shape_vertices.T).T
#         self.quadrature_weights = shape_volume * quadrature_reference_weights


# def get_quadrature(
#     shape_type: ShapeType,
#     integration_order: int,
#     choice=Choice,
#     quadrature_type: QuadratureType = QuadratureType.GAUSS,
# ) -> ndarray:
#     if quadrature_type == QuadratureType.GAUSS:
#         if shape_type == ShapeType.SEGMENT:
#             if not shape_vertices.shape[1] == 2:
#                 ValueError("wrong number of vertices")
#             quadrature_item = gauss_segment.get_segment_quadrature()
#             quadrature_fetcher = gauss_segment.get_segment_quadrature
#             quadrature_weights_fetcher = gauss_segment.get_segment_quadrature
#         elif shape_type == ShapeType.TRIANGLE:
#             if not shape_vertices.shape[1] == 3:
#                 ValueError("wrong number of vertices")
#         else:
#             raise KeyError("unsupported shape")
#     else:
#         raise KeyError("unsupported quadrature type")
