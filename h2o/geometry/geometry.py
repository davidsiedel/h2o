from h2o.h2o import *


def get_domain_barycenter(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    euclidean_dimension = vertices.shape[0]
    number_of_vertices = vertices.shape[1]
    shape_barycenter = np.zeros((euclidean_dimension,), dtype=real)
    for i in range(vertices.shape[1]):
        vertex = vertices[:, i]
        shape_barycenter += vertex
    shape_barycenter = (1.0 / number_of_vertices) * shape_barycenter
    return shape_barycenter


# def get_euclidean_norm(edge: ndarray) -> float:
#     """
#
#     Args:
#         edge:
#
#     Returns:
#
#     """
#     euclidean_norm = np.sqrt(np.sum(edge ** 2))
#     return euclidean_norm
