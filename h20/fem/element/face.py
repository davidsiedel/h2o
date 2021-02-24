from numpy import ndarray

from pythhon.parameters import *
from pythhon.geometry.shape import Shape
from pythhon.quadratures.shapequadrature import ShapeQuadrature


# e_0 = face_vertices_matrix[1, :] - face_vertices_matrix[0, :]
#             e_0 = e_0 / np.linalg.norm(e_0)
#             e_1 = np.array([e_0[1], -e_0[0]])
#             face_reference_frame_transformation_matrix = np.array([e_0, e_1])


def get_face_mapping_matrix(face_shape_type: ShapeType, face_vertices: ndarray) -> ndarray:
    """

    Args:
        face_shape_type:
        face_vertices:

    Returns:

    """
    if face_shape_type == ShapeType.POINT:
        mapping_matrix = np.eye(1)
    elif face_shape_type == ShapeType.SEGMENT:
        e_0 = face_vertices[:, 1] - face_vertices[:, 0]
        e_0 = e_0 / np.linalg.norm(e_0)
        e_1 = np.array([e_0[1], -e_0[0]])
        mapping_matrix = np.array([e_0, e_1])
    elif face_shape_type in [
        ShapeType.TRIANGLE,
        ShapeType.QUADRANGLE,
        ShapeType.POLYGON,
    ]:
        mapping_matrix = np.eye(100)
    else:
        raise KeyError("NO")
    return mapping_matrix


class Face:
    shape: Shape
    shape_type: ShapeType
    mapping_matrix: ndarray
    quadrature_points: ndarray
    quadrature_weights: ndarray

    def __init__(
        self,
        face_shape_type: ShapeType,
        face_vertices: ndarray,
        integration_order: int,
        quadrature_type: QuadratureType = QuadratureType.GAUSS,
    ):
        """

        Args:
            face_shape_type:
            face_vertices:
            integration_order:
            quadrature_type:
        """
        self.shape = Shape(face_shape_type, face_vertices)
        self.mapping_matrix = get_face_mapping_matrix(face_shape_type, face_vertices)
        # self.shape_type = face_shape_type
        face_quadrature = ShapeQuadrature(
            face_shape_type, face_vertices, self.shape.volume, integration_order, quadrature_type=quadrature_type,
        )
        self.quadrature_points = face_quadrature.quadrature_points
        self.quadrature_weights = face_quadrature.quadrature_weights
