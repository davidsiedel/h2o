from h2o.h2o import *
from h2o.fem.basis.basis import Basis


class FiniteElement:
    element_type: ElementType
    k_order: int
    construction_integration_order: int
    computation_integration_order: int
    basis_type: BasisType
    cell_basis_k: Basis
    cell_basis_l: Basis
    face_basis_k: Basis
    cell_basis_r: Basis

    def __init__(
        self,
        element_type: ElementType,
        polynomial_order: int,
        euclidean_dimension: int,
        basis_type: BasisType = BasisType.MONOMIAL,
    ):
        """

        Args:
            element_type:
            polynomial_order:
            euclidean_dimension:
            basis_type:
        """
        self.element_type = element_type
        self.basis_type = basis_type
        # --- POLYNOMIAL ORDERS
        self.k_order = polynomial_order
        self.r_order = polynomial_order + 1
        if element_type == ElementType.HDG_LOW:
            self.l_order = polynomial_order - 1
        elif element_type == ElementType.HDG_EQUAL:
            self.l_order = polynomial_order
        elif element_type == ElementType.HDG_HIGH:
            self.l_order = polynomial_order + 1
        else:
            raise ElementError("the specified element type is not known : {}".format(element_type))
        # --- BUILDING BASES
        self.cell_basis_k = Basis(self.k_order, euclidean_dimension, basis_type=basis_type)
        self.cell_basis_l = Basis(self.l_order, euclidean_dimension, basis_type=basis_type)
        self.cell_basis_r = Basis(self.r_order, euclidean_dimension, basis_type=basis_type)
        self.face_basis_k = Basis(self.k_order, euclidean_dimension - 1, basis_type=basis_type)
        # --- INTEGRATION ORDERS
        if debug_mode == DebugMode.LIGHT:
            self.construction_integration_order = 2 * (polynomial_order + 1)
        else:
            self.construction_integration_order = 2 * (polynomial_order + 1)
        self.computation_integration_order = 2 * (polynomial_order + 1)
        # self.computation_integration_order = 2 * (polynomial_order)
        # self.computation_integration_order = 2 * (polynomial_order)