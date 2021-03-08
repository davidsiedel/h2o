import numpy as np
from numpy import ndarray

import h2o.geometry.geometry as geom


def get_segment_barycenter(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    return geom.get_domain_barycenter(vertices)


def get_segment_diameter(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    diameter = get_segment_volume(vertices)
    return diameter


def get_segment_volume(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    edge = vertices[:, 0] - vertices[:, 1]
    volume = np.linalg.norm(edge)
    return volume


def get_segment_centroid(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    centroid = get_segment_barycenter(vertices)
    return centroid


# def get_segment_reference_frame_transformation_matrix(vertices: ndarray) -> ndarray:
#     """
#
#     :param vertices:
#     :return:
#     """
#     euclidean_dimension = vertices.shape[1]
#     vertices = vertices.reshape((vertices.shape[0], vertices.shape[1]))
#     if euclidean_dimension == 2:
#         # e_0 = vertices[1, :] - vertices[0, :]
#         e_0 = vertices[1] - vertices[0]
#         e_0 = e_0 / np.linalg.norm(e_0)
#         e_1 = np.array([e_0[1], -e_0[0]])
#         # e_1 = lnag.vector([e_0[1], -e_0[0]])
#         segment_reference_frame_transformation_matrix = np.array([e_0, e_1])
#         # segment_reference_frame_transformation_matrix = lnag.matrix([e_0, e_1])
#     elif euclidean_dimension == 1:
#         # segment_reference_frame_transformation_matrix = np.eye(1)
#         segment_reference_frame_transformation_matrix = Matrix(1, 1, ftype="EYE")
#     else:
#         raise EnvironmentError("wrong")
#     return segment_reference_frame_transformation_matrix
