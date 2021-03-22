from h2o.geometry.geometry import *
from h2o.geometry.shapes.shape_triangle import get_triangle_rotation_matrix


def get_quadrangle_barycenter(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    return get_domain_barycenter(vertices)


def get_quadrangle_diagonals(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    d_0 = vertices[:, 2] - vertices[:, 0]
    d_1 = vertices[:, 3] - vertices[:, 1]
    diagonals = np.array([d_0, d_1]).T
    return diagonals


def get_quadrangle_diameter(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    diagonals = get_quadrangle_diagonals(vertices)
    quadrangle_diameter = np.max([np.linalg.norm(diagonals[:,0]), np.linalg.norm(diagonals[:,1])])
    return quadrangle_diameter
    # edges = get_triangle_edges(vertices)
    # triangle_diameter = max([get_euclidean_norm(e) for e in edges])
    # return triangle_diameter


def get_quadrangle_volume(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    diagonals = get_quadrangle_diagonals(vertices)
    p = get_triangle_rotation_matrix(vertices)
    dprime = (p @ diagonals.T).T
    d_0 = dprime[:,0]
    d_1 = dprime[:,1]
    quadrangle_area = (1./2.) * np.abs(d_0[0] * d_1[1] - d_0[1] * d_1[0])
    # return quadrangle_area
    # d_0 = diagonals[:,0]
    # d_1 = diagonals[:,1]
    # quadrangle_area = (1./2.) * (d_0[0] * d_1[1] - d_0[1] * d_1[0])
    return quadrangle_area


def get_quadrangle_centroid(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    quadrangle_centroid = get_quadrangle_barycenter(vertices)
    return quadrangle_centroid


# def get_triangle_reference_frame_transformation_matrix(vertices: ndarray) -> ndarray:
#     """
#
#     Args:
#         vertices:
#
#     Returns:
#
#     """
#     euclidean_dimension = vertices.shape[0]
#     if euclidean_dimension == 3:
#         # e_0 = vertices[0] - vertices[-1]
#         # e_0 = vertices[2] - vertices[0]
#         e_0 = vertices[:, 2] - vertices[:, 0]
#         e_0 = e_0 / np.linalg.norm(e_0)
#         # e_t = vertices[1] - vertices[-1]
#         # e_t = vertices[1] - vertices[0]
#         e_t = vertices[:, 1] - vertices[:, 0]
#         e_t = e_t / np.linalg.norm(e_t)
#         e_2 = np.cross(e_0, e_t)
#         e_1 = np.cross(e_2, e_0)
#         triangle_reference_frame_transformation_matrix = np.array([e_0, e_1, e_2])
#     elif euclidean_dimension == 2:
#         triangle_reference_frame_transformation_matrix = np.eye(2)
#     else:
#         raise EnvironmentError("wrong")
#     return triangle_reference_frame_transformation_matrix
