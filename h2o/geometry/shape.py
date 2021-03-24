import h2o.geometry.shapes.shape_segment as isegment
import h2o.geometry.shapes.shape_triangle as itriangle
import h2o.geometry.shapes.shape_quadrangle as iquadrangle
import h2o.quadratures.quadrature as quad
from h2o.h2o import *

# def get_rotation_matrix(face_shape_type: ShapeType, face_vertices: ndarray) -> ndarray:
#     """
#     Args:
#         face_shape_type:
#         face_vertices:
#     Returns:
#     """
#     if face_shape_type == ShapeType.SEGMENT:
#         e_0 = face_vertices[:, 1] - face_vertices[:, 0]
#         e_0 = e_0 / np.linalg.norm(e_0)
#         e_1 = np.array([e_0[1], -e_0[0]])
#         mapping_matrix = np.array([e_0, e_1])
#     else:
#         raise KeyError("NO")
#     return mapping_matrix


def _check_shape(shape_type: ShapeType, shape_vertices: ndarray):
    """

    Args:
        shape_type:
        shape_vertices:

    Returns:

    """
    if shape_type == ShapeType.SEGMENT:
        if not shape_vertices.shape[1] == 2:
            raise ValueError("wrong number of vertices")
    elif shape_type == ShapeType.TRIANGLE:
        if not shape_vertices.shape[1] == 3:
            raise ValueError("wrong number of vertices")
    elif shape_type == ShapeType.QUADRANGLE:
        if not shape_vertices.shape[1] == 4:
            raise ValueError("wrong number of vertices")
    elif shape_type == ShapeType.TETRAHEDRON:
        if not shape_vertices.shape[1] == 4:
            raise ValueError("wrong number of vertices")
    elif shape_type == ShapeType.HEXAHEDRON:
        if not shape_vertices.shape[1] == 8:
            raise ValueError("wrong number of vertices")
    else:
        raise KeyError("unsupported shape")


class Shape:
    type: ShapeType
    vertices: ndarray
    centroid: ndarray
    volume: float
    diameter: float

    def __init__(self, shape_type: ShapeType, shape_vertices: ndarray):
        """

        Args:
            shape_type:
            shape_vertices:
        """
        _check_shape(shape_type, shape_vertices)
        self.type = shape_type
        self.vertices = shape_vertices
        if shape_type == ShapeType.SEGMENT:
            self.centroid = isegment.get_segment_centroid(shape_vertices)
            self.volume = isegment.get_segment_volume(shape_vertices)
            self.diameter = isegment.get_segment_diameter(shape_vertices)
        elif shape_type == ShapeType.TRIANGLE:
            self.centroid = itriangle.get_triangle_centroid(shape_vertices)
            self.volume = itriangle.get_triangle_volume(shape_vertices)
            self.diameter = itriangle.get_triangle_diameter(shape_vertices)
        elif shape_type == ShapeType.QUADRANGLE:
            self.centroid = iquadrangle.get_quadrangle_centroid(shape_vertices)
            self.volume = iquadrangle.get_quadrangle_volume(shape_vertices)
            self.diameter = iquadrangle.get_quadrangle_diameter(shape_vertices)

    def get_quadrature_points(
        self, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
    ) -> ndarray:
        """

        Args:
            integration_order:
            quadrature_type:

        Returns:

        """
        quadrature_points = quad.get_shape_quadrature_points(
            self.type, self.vertices, integration_order, quadrature_type=quadrature_type
        )
        return quadrature_points

    def get_quadrature_weights(
        self, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
    ) -> ndarray:
        """

        Args:
            integration_order:
            quadrature_type:

        Returns:

        """
        # quadrature_weights = quad.get_shape_quadrature_weights(
        #     self.type, self.volume, integration_order, quadrature_type=quadrature_type
        # )
        quadrature_weights = quad.get_shape_quadrature_weights(
            self.type, self.vertices, integration_order, quadrature_type=quadrature_type
        )
        return quadrature_weights

    def get_quadrature_size(
        self, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
    ) -> int:
        """

        Args:
            integration_order:
            quadrature_type:

        Returns:

        """
        quadrature_size = quad.get_shape_quadrature_size(
            self.type, integration_order, quadrature_type=quadrature_type
        )
        return quadrature_size
