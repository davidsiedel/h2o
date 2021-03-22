from h2o.geometry.geometry import *
from scipy.special import binom
from h2o.geometry.shapes.shape_triangle import get_triangle_edges


def get_polygon_barycenter(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    return get_domain_barycenter(vertices)


# def get_triangle_edges(vertices: ndarray) -> ndarray:
#     """
#
#     Args:
#         vertices:
#
#     Returns:
#
#     """
#     # e_0 = vertices[1, :] - vertices[0, :]
#     # e_1 = vertices[2, :] - vertices[1, :]
#     # e_2 = vertices[0, :] - vertices[2, :]
#     e_0 = vertices[:, 1] - vertices[:, 0]
#     e_1 = vertices[:, 2] - vertices[:, 1]
#     e_2 = vertices[:, 0] - vertices[:, 2]
#     edges = np.array([e_0, e_1, e_2])
#     return edges

def get_polygon_partition(vertices: ndarray) -> ndarray:
    number_of_vertices = vertices.shape[1]
    euclidean_dimension = vertices.shape[0]
    polygon_centroid = get_polygon_centroid(vertices)
    # simplicial_sub_domains = []
    simplicial_sub_domains = np.zeros((number_of_vertices, euclidean_dimension, 3), dtype=real)
    for i in range(number_of_vertices):
        simplicial_sub_domains[i] = np.array(
            [
                vertices[:, i - 1],
                vertices[:, i],
                polygon_centroid,
            ], dtype=real
        ).T
        # sub_domain_vertices = [
        #     vertices[:, i - 1],
        #     vertices[:, i],
        #     polygon_centroid,
        # ]
        # simplicial_sub_domains.append(np.array(sub_domain_vertices))
    return simplicial_sub_domains


def get_polygon_diameter(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    number_of_vertices = vertices.shape[1]
    # number_of_combinations = binom(number_of_vertices, 2)
    # --------------------------------------------------------------------------------------------------------------
    # combinations_count = 0
    # lengths = []
    polygon_diameter = 0.0
    for i in range(number_of_vertices):
        v0 = vertices[:, i]
        for j in range(number_of_vertices):
            v1 = vertices[:, j]
            # if not i == j and not permutation_count == number_of_combinations:
            if not i == j:
                edge = v1 - v0
                edge_length = np.linalg.norm(edge)
                if edge_length > polygon_diameter:
                    polygon_diameter = edge_length
                # lengths.append(np.sqrt((e[1] + e[0]) ** 2))
    # polygon_diameter = max(lengths)
    return polygon_diameter

def get_lace(vertices: ndarray, index: int) -> float:
    """

    Args:
        vertices:
        index:

    Returns:

    """
    lace = vertices[0, index - 1] * vertices[1, index] - vertices[0, index] * vertices[1, index - 1]
    return lace

def get_polygon_volume(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    number_of_vertices = vertices.shape[1]
    # shoe_lace_matrix = []
    lace_sum = 0.0
    for i in range(number_of_vertices):
        # lace = vertices[i - 1][0] * vertices[i][1] - vertices[i][0] * vertices[i - 1][1]
        # lace = vertices[0, i - 1] * vertices[1, i] - vertices[0, i] * vertices[1, i - 1]
        lace = get_lace(vertices, i)
        # shoe_lace_matrix.append(lace)
        lace_sum += lace
    # polygon_volume = np.abs(1.0 / 2.0 * np.sum(shoe_lace_matrix))
    polygon_volume = np.abs(1.0 / 2.0 * lace_sum)
    # shoe_lace_matrix = []
    # for i in range(number_of_vertices):
    #     edge_matrix = np.array([vertices[i - 1], vertices[i]])
    #     lace = np.linalg.det(edge_matrix.T)
    #     shoe_lace_matrix.append(lace)
    # polygon_volume = np.abs(1.0 / 2.0 * np.sum(shoe_lace_matrix))
    return polygon_volume


def get_polygon_centroid(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    number_of_vertices = vertices.shape[1]
    polygon_volume = get_polygon_volume(vertices)
    # centroid_matrix_x = []
    cx_sum = 0.0
    for i in range(number_of_vertices):
        # centroid_matrix_x.append(
        #     (vertices[i - 1][0] + vertices[i][0])
        #     * (vertices[i - 1][0] * vertices[i][1] - vertices[i][0] * vertices[i - 1][1])
        # )
        cx_sum += (vertices[0, i - 1] + vertices[0, i]) * get_lace(vertices, i)
    # polygon_centroid_x = 1.0 / (6.0 * polygon_volume) * np.sum(centroid_matrix_x)
    polygon_centroid_x = 1.0 / (6.0 * polygon_volume) * cx_sum
    # centroid_matrix_y = []
    cy_sum = 0.0
    for i in range(number_of_vertices):
        # centroid_matrix_y.append(
        #     (vertices[i - 1][1] + vertices[i][1])
        #     * (vertices[i - 1][0] * vertices[i][1] - vertices[i][0] * vertices[i - 1][1])
        # )
        cy_sum += (vertices[1, i - 1] + vertices[1, i]) * get_lace(vertices, i)
    # polygon_centroid_y = 1.0 / (6.0 * polygon_volume) * np.sum(centroid_matrix_y)
    polygon_centroid_y = 1.0 / (6.0 * polygon_volume) * cy_sum
    polygon_centroid = np.array([polygon_centroid_x, polygon_centroid_y])
    return polygon_centroid


def get_polygon_rotation_matrix(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    polygon_partition = get_polygon_partition(vertices)
    euclidean_dimension = vertices.shape[0]
    for triangle_vertices in polygon_partition:
        triangle_edges = get_triangle_edges(triangle_vertices)
        e0 = triangle_edges[:, 0] / np.linalg.norm(triangle_edges[:, 0])
        e1 = triangle_edges[:, 1] / np.linalg.norm(triangle_edges[:, 1])
        # cos_check = np.tensordot(triangle_edges[:, 0], triangle_edges[:, 1], axes=2)
        cos_check = triangle_edges[:, 0] @ triangle_edges[:, 1]
        if cos_check != 1.0 and cos_check != -1.0:
            # euclidean_dimension = vertices.shape[0]
            if euclidean_dimension == 3:
                # e_0 = vertices[0] - vertices[-1]
                # e_0 = vertices[2] - vertices[0]
                e_0 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
                e_0 = e_0 / np.linalg.norm(e_0)
                # e_t = vertices[1] - vertices[-1]
                # e_t = vertices[1] - vertices[0]
                e_t = triangle_vertices[:, 1] - triangle_vertices[:, 0]
                e_t = e_t / np.linalg.norm(e_t)
                e_2 = np.cross(e_0, e_t)
                e_1 = np.cross(e_2, e_0)
                triangle_reference_frame_transformation_matrix = np.array([e_0, e_1, e_2])
            elif euclidean_dimension == 2:
                triangle_reference_frame_transformation_matrix = np.eye(2)
            else:
                raise EnvironmentError("wrong")
            return triangle_reference_frame_transformation_matrix
        else:
            pass
    raise ValueError("all points in the polygon are aligned")
