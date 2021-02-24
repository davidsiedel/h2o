from numpy import ndarray as ndarray
from typing import List, Dict
import pythhon.mesh.parsers.geof as geof
from pythhon.quadratures.shapequadrature import ShapeQuadrature
from pythhon.parameters import *


def get_number_of_quadrature_points_in_mesh(
    items_shapes: List[ShapeType], integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
) -> int:
    """

    Args:
        items_shapes: a list enumerating the shape of the items passed to the function, either cells or faces
        integration_order: the polynomial integration order, depending on the finite element order
        quadrature_type: the type of quadrature used to compute quadrature points and weights

    Returns: the number of quadrature points in the Mesh, for the considered item (either the mesh or its skeleton)

    """
    number_of_quadrature_points_in_mesh = 0
    for item_shape in items_shapes:
        n = ShapeQuadrature.get_number_of_quadrature_points(
            item_shape, integration_order, quadrature_type=quadrature_type
        )
        number_of_quadrature_points_in_mesh += n
    return number_of_quadrature_points_in_mesh


class Mesh:
    vertices: ndarray
    euclidean_dimension: int
    cells_vertices_connectivity: List[List[int]]
    cells_shape_types: List[ShapeType]
    faces_vertices_connectivity: List[List[int]]
    faces_shape_types: List[ShapeType]
    cells_faces_connectivity: List[List[int]]
    vertices_boundaries_connectivity: Dict[str, List[int]]
    faces_boundaries_connectivity: Dict[str, List[int]]
    number_of_vertices_in_mesh: int
    number_of_cells_in_mesh: int
    number_of_faces_in_mesh: int
    number_of_face_quadrature_points_in_mesh: int
    number_of_cell_quadrature_points_in_mesh: int
    vertices_weights_cell: ndarray
    vertices_weights_face: ndarray

    def __init__(
        self, mesh_file_path: str, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
    ):
        """

        Args:
            mesh_file_path: the file path to the mesh to read
            integration_order: the polynomial integration order, depending on the finite element order
            quadrature_type: the type of quadrature used to compute quadrature points and weights
        """
        if ".geof" in mesh_file_path:
            vertices = geof.get_vertices(mesh_file_path)
            (cells_vertices_connectivity, cells_shapes, cells_labels) = geof.get_cells_data(mesh_file_path)
            (faces_vertices_connectivity, cells_faces_connectivity, faces_shapes) = geof.get_faces_data(
                cells_vertices_connectivity, cells_labels
            )
            vertices_boundaries_connectivity = geof.get_vertices_boundaries_connectivity(mesh_file_path)
            faces_boundaries_connectivity = geof.get_faces_boundaries_connectivity(
                vertices_boundaries_connectivity, faces_vertices_connectivity
            )
            self.vertices = vertices
            self.euclidean_dimension = self.vertices.shape[0]
            self.cells_vertices_connectivity = cells_vertices_connectivity
            self.cells_shape_types = cells_shapes
            self.faces_vertices_connectivity = faces_vertices_connectivity
            self.faces_shape_types = faces_shapes
            self.cells_faces_connectivity = cells_faces_connectivity
            self.vertices_boundaries_connectivity = vertices_boundaries_connectivity
            self.faces_boundaries_connectivity = faces_boundaries_connectivity
            self.number_of_vertices_in_mesh = self.get_number_of_vertices_in_mesh()
            self.number_of_cells_in_mesh = self.get_number_of_cells_in_mesh()
            self.number_of_faces_in_mesh = self.get_number_of_faces_in_mesh()
            self.number_of_cell_quadrature_points_in_mesh = get_number_of_quadrature_points_in_mesh(
                cells_shapes, integration_order, quadrature_type=quadrature_type
            )
            self.number_of_face_quadrature_points_in_mesh = get_number_of_quadrature_points_in_mesh(
                faces_shapes, integration_order, quadrature_type=quadrature_type
            )
            self.vertices_weights_cell = geof.get_vertices_weights(mesh_file_path, cells_vertices_connectivity)
            self.vertices_weights_face = geof.get_vertices_weights(mesh_file_path, faces_vertices_connectivity)
        else:
            raise IOError("unsupported mesh file format")

    def get_number_of_cells_in_mesh(self) -> int:
        """

        Returns: the number of cells in the mesh

        """
        number_of_cells_in_mesh = len(self.cells_vertices_connectivity)
        return number_of_cells_in_mesh

    def get_number_of_faces_in_mesh(self) -> int:
        """

        Returns: the number of faces in the mesh

        """
        number_of_faces_in_mesh = len(self.faces_vertices_connectivity)
        return number_of_faces_in_mesh

    def get_number_of_vertices_in_mesh(self) -> int:
        """

        Returns: the number of vertices in the mesh

        """
        number_of_vertices_in_mesh = self.vertices.shape[1]
        return number_of_vertices_in_mesh


# def get_number_of_cell_quadrature_points_in_mesh(
#     vertices: ndarray,
#     cells_vertices_connectivity: List[int],
#     integration_order: int,
#     quadrature_type: QuadratureType = QuadratureType.GAUSS,
# ) -> int:
#     """
#
#     :param vertices:
#     :param cells_vertices_connectivity:
#     :param integration_order:
#     :param quadrature_type:
#     :return:
#     """
#     number_of_cell_quadrature_points_in_mesh = 0
#     for cell_vertices_indices in cells_vertices_connectivity:
#         cell_vertices = ishape.get_shape_vertices(vertices, cell_vertices_indices)
#         cell_shape_type = get_cell_shape_type(cell_vertices)
#         n = Quadrature.get_number_of_quadrature_points(
#             integration_order, cell_shape_type, quadrature_type=quadrature_type
#         )
#         number_of_cell_quadrature_points_in_mesh += n
#     return number_of_cell_quadrature_points_in_mesh
#
#
# def get_cell_quadrature_in_mesh(
#     vertices: ndarray,
#     cells_vertices_connectivity_matrix: List[int],
#     integration_order: int,
#     quadrature_type: QuadratureType = QuadratureType.GAUSS,
# ) -> (ndarray, ndarray, ndarray):
#     """
#
#     :param vertices:
#     :param cells_vertices_connectivity_matrix:
#     :param integration_order:
#     :param quadrature_type:
#     :return:
#     """
#     euclidean_dimension = vertices.shape[0]
#     nc = get_number_of_cell_quadrature_points_in_mesh(
#         cells_vertices_connectivity_matrix,
#         integration_order,
#         quadrature_type=quadrature_type,
#     )
#     cell_quadrature_points = np.zeros((euclidean_dimension, nc))
#     cell_quadrature_weights = np.zeros((nc,))
#     number_of_cells_in_mesh = get_number_of_cells_in_mesh(
#         cells_vertices_connectivity_matrix
#     )
#     cells_quadrature_points_range_matrix = np.zeros(
#         (number_of_cells_in_mesh, 2), dtype=np.uint8
#     )
#     qp_count = 0
#     for cell_index, cell_vertices_indices in enumerate(
#         cells_vertices_connectivity_matrix
#     ):
#         cell_vertices = ishape.get_shape_vertices(vertices, cell_vertices_indices)
#         cell_shape_type = get_cell_shape_type(cell_vertices)
#         (
#             cell_quadrature_points,
#             cell_quadrature_weights,
#         ) = ishape.get_shape_quadrature(
#             cell_vertices,
#             cell_shape_type,
#             integration_order,
#             quadrature_type=quadrature_type,
#         )
#         nq = Quadrature.get_number_of_quadrature_points(
#             integration_order, cell_shape_type, quadrature_type=quadrature_type
#         )
#         cell_quadrature_points[:, qp_count : qp_count + nq] += cell_quadrature_points
#         cell_quadrature_weights[qp_count : qp_count + nq] += cell_quadrature_weights
#         cells_quadrature_points_range_matrix[cell_index, 0] = qp_count
#         cells_quadrature_points_range_matrix[cell_index, 1] = qp_count + nq
#         qp_count += nq
#     return (
#         cell_quadrature_points,
#         cell_quadrature_weights,
#         cells_quadrature_points_range_matrix,
#     )
#
#
# def get_face_quadrature_in_mesh(
#     vertices: ndarray,
#     faces_vertices_connectivity_matrix: List[int],
#     integration_order: int,
#     quadrature_type: QuadratureType = QuadratureType.GAUSS,
# ) -> (ndarray, ndarray, ndarray):
#     """
#
#     :param vertices:
#     :param faces_vertices_connectivity_matrix:
#     :param integration_order:
#     :param quadrature_type:
#     :return:
#     """
#     euclidean_dimension = vertices.shape[1]
#     nc = get_number_of_face_quadrature_points_in_mesh(
#         faces_vertices_connectivity_matrix,
#         integration_order,
#         quadrature_type=quadrature_type,
#     )
#     face_quadrature_points = np.zeros((nc, euclidean_dimension))
#     face_quadrature_weights = np.zeros((nc,))
#     number_of_faces_in_mesh = get_number_of_faces_in_mesh(
#         faces_vertices_connectivity_matrix
#     )
#     faces_quadrature_points_range_matrix = np.zeros(
#         (number_of_faces_in_mesh, 2), dtype=np.uint8
#     )
#     qp_count = 0
#     for face_index, face_vertices_indices in enumerate(
#         faces_vertices_connectivity_matrix
#     ):
#         face_vertices = ishape.get_shape_vertices(vertices, face_vertices_indices)
#         face_shape_type = get_face_shape_type(face_vertices)
#         (
#             face_quadrature_points,
#             face_quadrature_weights,
#         ) = ishape.get_shape_quadrature(
#             face_vertices,
#             face_shape_type,
#             integration_order,
#             quadrature_type=quadrature_type,
#         )
#         nq = Quadrature.get_number_of_quadrature_points(
#             integration_order, face_shape_type, quadrature_type=quadrature_type
#         )
#         face_quadrature_points[qp_count : qp_count + nq] += face_quadrature_points
#         face_quadrature_weights[qp_count : qp_count + nq] += face_quadrature_weights
#         faces_quadrature_points_range_matrix[face_index, 0] = qp_count
#         faces_quadrature_points_range_matrix[face_index, 1] = qp_count + nq
#         qp_count += nq
#     return (
#         face_quadrature_points,
#         face_quadrature_weights,
#         faces_quadrature_points_range_matrix,
#     )
#
#
# def get_quadrature_points_connectivity_matrix(
#     quadrature_points_range_matrix: ndarray, item_index: int
# ) -> ndarray:
#     """
#
#     :param quadrature_points_range_matrix:
#     :param item_index:
#     :return:
#     """
#     start = quadrature_points_range_matrix[item_index, 0]
#     stop = quadrature_points_range_matrix[item_index, 1]
#     quadrature_points_connectivity_matrix = np.arange(start, stop, dtype=np.uint8)
#     return quadrature_points_connectivity_matrix
