import numpy as np
from numpy import ndarray as matrix
from typing import List, Callable, Dict


def get_number_of_quadrature_points_in_point() -> int:
    """

    Returns:

    """
    return 1


def get_point_quadrature_data() -> (matrix, matrix):
    """

    Returns:

    """
    reference_points = np.array([[1.0]])
    reference_weights = np.array([1.0])
    # quadrature_points, quadrature_weights = np.array([[]]), np.array([[1.0]])
    # quadrature_points = reference_points @ vertices
    # quadrature_weights = reference_weights * volume
    # return quadrature_points, quadrature_weights
    return reference_points, reference_weights
