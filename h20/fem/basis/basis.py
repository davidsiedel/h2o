from numpy import ndarray
from typing import Callable

import pythhon.fem.basis.bases.monomial as imonomial
from pythhon.parameters import *


class Basis:
    dimension: int
    evaluate_derivative: Callable[[ndarray, ndarray, float, int], ndarray]
    evaluate_function: Callable[[ndarray, ndarray, float], ndarray]

    def __init__(
        self, polynomial_order: int, euclidean_dimension: int, basis_type: BasisType = BasisType.MONOMIAL,
    ):
        """

        Args:
            polynomial_order:
            euclidean_dimension:
            basis_type:
        """
        if basis_type == BasisType.MONOMIAL:
            b = imonomial.Monomial(polynomial_order, euclidean_dimension)
        else:
            raise KeyError("unsupported basis")
        self.dimension = b.dimension
        self.evaluate_function = b.get_phi_vector
        self.evaluate_derivative = b.get_d_phi_vector
