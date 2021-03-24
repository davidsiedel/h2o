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
        self.construction_integration_order = 2 * (polynomial_order + 1)
        self.computation_integration_order = 2 * (polynomial_order + 1)
        #     self.cell_basis_k = Basis(
        #         self.k_order, euclidean_dimension, basis_type=basis_type
        #     )
        #     self.cell_basis_l = Basis(
        #         self.l_order, euclidean_dimension, basis_type=basis_type
        #     )
        #     self.face_basis_k = Basis(
        #         self.k_order, euclidean_dimension - 1, basis_type=basis_type
        #     )
        #     self.cell_basis_r = Basis(
        #         self.r_order, euclidean_dimension, basis_type=basis_type
        #     )
        #     self.construction_integration_order = 2 * (polynomial_order + 1)
        #     self.computation_integration_order = 2 * (polynomial_order + 1)
        # elif element_type == ElementType.HDG_EQUAL:
        #     self.l_order = polynomial_order
        #     self.cell_basis_k = Basis(
        #         polynomial_order, euclidean_dimension, basis_type=basis_type
        #     )
        #     self.cell_basis_l = Basis(
        #         self.l_order, euclidean_dimension, basis_type=basis_type
        #     )
        #     self.face_basis_k = Basis(
        #         polynomial_order, euclidean_dimension - 1, basis_type=basis_type
        #     )
        #     self.cell_basis_r = Basis(
        #         polynomial_order + 1, euclidean_dimension, basis_type=basis_type
        #     )
        #     self.construction_integration_order = 2 * (polynomial_order + 1)
        #     self.computation_integration_order = 2 * (polynomial_order + 1)
        # elif element_type == ElementType.HDG_HIGH:
        #     self.l_order = polynomial_order + 1
        #     self.cell_basis_k = Basis(
        #         polynomial_order, euclidean_dimension, basis_type=basis_type
        #     )
        #     self.cell_basis_l = Basis(
        #         self.l_order, euclidean_dimension, basis_type=basis_type
        #     )
        #     self.face_basis_k = Basis(
        #         polynomial_order, euclidean_dimension - 1, basis_type=basis_type
        #     )
        #     self.cell_basis_r = Basis(
        #         polynomial_order + 1, euclidean_dimension, basis_type=basis_type
        #     )
        #     self.construction_integration_order = 2 * (polynomial_order + 1)
        #     self.computation_integration_order = 2 * (polynomial_order + 1)
        #
        # else:
        #     raise KeyError("NO")
        # return

        # if self.field.field_type == FieldType.SCALAR:
        #     # self.gradient_type = DerivationType.FULL
        #     self.field_dimension = 1
        #     self.gradient_dimension = euclidean_dimension
        #     self.voigt_indices = {
        #         (0, 0): 0,
        #         (0, 1): 1
        #     }
        #     self.voigt_coefficients = {
        #         (0, 0): 1.0,
        #         (0, 1): 1.0
        #     }
        # elif self.field.field_type == FieldType.VECTOR:
        #     self.gradient_type = DerivationType.FULL
        #     self.field_dimension = euclidean_dimension
        #     self.gradient_dimension = euclidean_dimension * euclidean_dimension
        #     if euclidean_dimension == 2:
        #         self.voigt_indices = {
        #             (0, 0): 0,
        #             (1, 1): 1,
        #             (0, 1): 2,
        #             (1, 0): 3
        #         }
        #         self.voigt_coefficients = {
        #             (0, 0): 1.0,
        #             (1, 1): 1.0,
        #             (0, 1): 1.0,
        #             (1, 0): 1.0,
        #         }
        #     elif euclidean_dimension == 3:
        #         self.voigt_indices = {
        #             (0, 0): 0,
        #             (1, 1): 1,
        #             (2, 2): 2,
        #             (0, 1): 3,
        #             (1, 0): 4,
        #             (0, 2): 5,
        #             (2, 0): 6,
        #             (1, 2): 7,
        #             (2, 1): 8,
        #         }
        #         self.voigt_coefficients = {
        #             (0, 0): 1.0,
        #             (1, 1): 1.0,
        #             (2, 2): 1.0,
        #             (0, 1): 1.0,
        #             (1, 0): 1.0,
        #             (0, 2): 1.0,
        #             (2, 0): 1.0,
        #             (1, 2): 1.0,
        #             (2, 1): 1.0,
        #         }
        # elif self.field_type == FieldType.DISPLACEMENT_FINITE_STRAIN:
        #     self.gradient_type = DerivationType.FULL
        #     self.field_dimension = euclidean_dimension
        #     self.gradient_dimension = euclidean_dimension * euclidean_dimension
        #     if euclidean_dimension == 2:
        #         self.voigt_indices = {
        #             (0, 0): 0,
        #             (1, 1): 1,
        #             (0, 1): 2,
        #             (1, 0): 3
        #         }
        #         self.voigt_coefficients = {
        #             (0, 0): 1.0,
        #             (1, 1): 1.0,
        #             (0, 1): 1.0,
        #             (1, 0): 1.0,
        #         }
        #     elif euclidean_dimension == 3:
        #         self.voigt_indices = {
        #             (0, 0): 0,
        #             (1, 1): 1,
        #             (2, 2): 2,
        #             (0, 1): 3,
        #             (1, 0): 4,
        #             (0, 2): 5,
        #             (2, 0): 6,
        #             (1, 2): 7,
        #             (2, 1): 8,
        #         }
        #         self.voigt_coefficients = {
        #             (0, 0): 1.0,
        #             (1, 1): 1.0,
        #             (2, 2): 1.0,
        #             (0, 1): 1.0,
        #             (1, 0): 1.0,
        #             (0, 2): 1.0,
        #             (2, 0): 1.0,
        #             (1, 2): 1.0,
        #             (2, 1): 1.0,
        #         }
        # elif self.field_type == FieldType.DISPLACEMENT_SMALL_STRAIN:
        #     self.gradient_type = DerivationType.SYMMETRIC
        #     self.field_dimension = euclidean_dimension
        #     self.gradient_dimension = euclidean_dimension * euclidean_dimension
        #     if euclidean_dimension == 2:
        #         self.voigt_indices = {
        #             (0, 0): 0,
        #             (1, 1): 1,
        #             (0, 1): 2
        #         }
        #         self.voigt_coefficients = {
        #             (0, 0): 1.0,
        #             (1, 1): 1.0,
        #             (0, 1): np.sqrt(2.0)
        #         }
        #     elif euclidean_dimension == 3:
        #         self.voigt_indices = {
        #             (0, 0): 0,
        #             (1, 1): 1,
        #             (2, 2): 2,
        #             (0, 1): 3,
        #             (0, 2): 4,
        #             (1, 2): 5,
        #         }
        #         self.voigt_coefficients = {
        #             (0, 0): 1.0,
        #             (1, 1): 1.0,
        #             (2, 2): 1.0,
        #             (0, 1): np.sqrt(2.0),
        #             (0, 2): np.sqrt(2.0),
        #             (1, 2): np.sqrt(2.0),
        #         }
        # elif (
        #     self.field_type == FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN
        #     or self.field_type == FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRESS
        # ):
        #     self.gradient_type = DerivationType.SYMMETRIC
        #     if self.euclidean_dimension != 2:
        #         raise ValueError("The euclidean dimension must be 2 !")
        #     self.field_dimension = 2
        #     self.gradient_dimension = 4
        #     self.voigt_indices = {
        #         (0, 0): 0,
        #         (1, 1): 1,
        #         (0, 1): 3
        #     }
        #     self.voigt_coefficients = {
        #         (0, 0): 1.0,
        #         (1, 1): 1.0,
        #         (0, 1): np.sqrt(2.0)
        #     }
        # elif (
        #     self.field_type == FieldType.DISPLACEMENT_FINITE_STRAIN_PLANE_STRAIN
        #     or self.field_type == FieldType.DISPLACEMENT_FINITE_STRAIN_PLANE_STRESS
        # ):
        #     self.gradient_type = DerivationType.FULL
        #     if self.euclidean_dimension != 2:
        #         raise ValueError("The euclidean dimension must be 2 !")
        #     self.field_dimension = 2
        #     self.gradient_dimension = 5
        #     self.voigt_indices = {
        #         (0, 0): 0,
        #         (1, 1): 1,
        #         (0, 1): 3,
        #         (1, 0): 4
        #     }
        #     self.voigt_coefficients = {
        #         (0, 0): 1.0,
        #         (1, 1): 1.0,
        #         (0, 1): 1.0,
        #         (1, 0): 1.0,
        #     }
        # else:
        #     raise KeyError("NO")
