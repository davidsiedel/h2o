import numpy as np
from numpy import ndarray
from typing import List, Callable, Dict

from pythhon.geometry.geometry import *


def get_triangle_barycenter(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    return get_domain_barycenter(vertices)


def get_triangle_edges(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    # e_0 = vertices[1, :] - vertices[0, :]
    # e_1 = vertices[2, :] - vertices[1, :]
    # e_2 = vertices[0, :] - vertices[2, :]
    e_0 = vertices[:, 1] - vertices[:, 0]
    e_1 = vertices[:, 2] - vertices[:, 1]
    e_2 = vertices[:, 0] - vertices[:, 2]
    edges = np.array([e_0, e_1, e_2])
    return edges


def get_triangle_diameter(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    edges = get_triangle_edges(vertices)
    triangle_diameter = max([get_euclidean_norm(e) for e in edges])
    return triangle_diameter


def get_triangle_volume(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    edges = get_triangle_edges(vertices)
    p = get_triangle_reference_frame_transformation_matrix(vertices)
    e0 = (p @ edges.T).T
    triangle_volume = np.abs(1.0 / 2.0 * np.linalg.det(e0[0:2, 0:2]))
    return triangle_volume


def get_triangle_centroid(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    triangle_centroid = get_triangle_barycenter(vertices)
    return triangle_centroid


def get_triangle_reference_frame_transformation_matrix(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    euclidean_dimension = vertices.shape[0]
    if euclidean_dimension == 3:
        # e_0 = vertices[0] - vertices[-1]
        # e_0 = vertices[2] - vertices[0]
        e_0 = vertices[:, 2] - vertices[:, 0]
        e_0 = e_0 / np.linalg.norm(e_0)
        # e_t = vertices[1] - vertices[-1]
        # e_t = vertices[1] - vertices[0]
        e_t = vertices[:, 1] - vertices[:, 0]
        e_t = e_t / np.linalg.norm(e_t)
        e_2 = np.cross(e_0, e_t)
        e_1 = np.cross(e_2, e_0)
        triangle_reference_frame_transformation_matrix = np.array([e_0, e_1, e_2])
    elif euclidean_dimension == 2:
        triangle_reference_frame_transformation_matrix = np.eye(2)
    else:
        raise EnvironmentError("wrong")
    return triangle_reference_frame_transformation_matrix
