from numpy import ndarray as ndarray
from typing import List, Dict
import h2o.mesh.parsers.geof as geof
from h2o.quadratures.quadrature import ShapeQuadrature
from h2o.h2o import *


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
        n = get_number_of_quadrature_points(
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