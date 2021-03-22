from random import uniform
from unittest import TestCase

import matplotlib.pyplot as plt
import quadpy

from h2o.fem.basis.bases.monomial import Monomial
from h2o.geometry.shape import Shape, get_rotation_matrix
from h2o.fem.element.finite_element import FiniteElement
from h2o.h2o import *

np.set_printoptions(precision=16)
np.set_printoptions(linewidth=1)


class TestCellQuadrangle(TestCase):
    def test_cell_quadrangle(self, verbose=False):

        euclidean_dimension = 2
        polynomial_orders = [1, 2, 3]
        element_types = [ElementType.HDG_LOW, ElementType.HDG_EQUAL, ElementType.HDG_HIGH]
        for polynomial_order in polynomial_orders:
            for element_type in element_types:
                # --------------------------------------------------------------------------------------------------------------
                # DEFINE POLYNOMIAL ORDER AND INTEGRATION ORDER
                # --------------------------------------------------------------------------------------------------------------
                finite_element = FiniteElement(
                    element_type=element_type,
                    polynomial_order=polynomial_order,
                    euclidean_dimension=euclidean_dimension,
                )

                # --------------------------------------------------------------------------------------------------------------
                # DEFINE POLYNOMIAL BASIS
                # --------------------------------------------------------------------------------------------------------------
                cell_basis_k = finite_element.cell_basis_k
                cell_basis_l = finite_element.cell_basis_l

                # --------------------------------------------------------------------------------------------------------------
                # DEFINE RANDOM POLYNOMIAL COEFFICIENTS
                # --------------------------------------------------------------------------------------------------------------
                range_min = -3.0
                range_max = +3.0
                coefficients_k = np.array([uniform(range_min, range_max) for _i in range(cell_basis_k.dimension)])
                coefficients_l = np.array([uniform(range_min, range_max) for _i in range(cell_basis_l.dimension)])
                if verbose == True:
                    print("COEFS_K : \n{}".format(coefficients_k))
                    print("COEFS_L : \n{}".format(coefficients_l))

                # --------------------------------------------------------------------------------------------------------------
                # DEFINE MONOMIAL VALUES COMPUTATION
                # --------------------------------------------------------------------------------------------------------------
                def test_function(
                    polynomial_ord: int, point: ndarray, centroid: ndarray, diameter: float, coefficients: ndarray
                ) -> float:
                    basis = Monomial(polynomial_ord, euclidean_dimension)
                    value = 0.0
                    for _i, _exponent in enumerate(basis.exponents):
                        prod = 1.0
                        for _x_dir in range(basis.exponents.shape[1]):
                            prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
                        prod *= coefficients[_i]
                        value += prod
                    return value

                def test_function_derivative(
                    polynomial_ord: int,
                    point: ndarray,
                    centroid: ndarray,
                    diameter: float,
                    direction: int,
                    coefficients: ndarray,
                ) -> float:
                    basis = Monomial(polynomial_ord, euclidean_dimension)
                    value = 0.0
                    for _i, _exponent in enumerate(basis.exponents):
                        prod = 1.0
                        for _x_dir in range(basis.exponents.shape[1]):
                            if _x_dir == direction:
                                _pt0 = point[_x_dir] - centroid[_x_dir]
                                _pt1 = _pt0 / diameter
                                if _exponent[_x_dir] == 0:
                                    _exp = _exponent[_x_dir]
                                else:
                                    _exp = _exponent[_x_dir] - 1
                                _pt2 = _pt1 ** _exp
                                # prod *= (_exponent[_x_dir] / diameter) * (
                                #         ((point[_x_dir] - centroid[_x_dir]) / diameter) ** (_exponent[_x_dir] - 1)
                                # )
                                prod *= (_exponent[_x_dir] / diameter) * _pt2
                            else:
                                prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
                        prod *= coefficients[_i]
                        value += prod
                    return value

                # def get_phi_vector(point: ndarray, centroid: ndarray, diameter: float) -> ndarray:
                #     phi_vector = np.zeros((cell_basis_k.dimension,), dtype=real)
                #     for _i, _exponent in enumerate(cell_basis_k.exponents):
                #         prod = 1.0
                #         for _x_dir in range(cell_basis_k.exponents.shape[1]):
                #             prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
                #         phi_vector[_i] += prod
                #     return phi_vector

                # --------------------------------------------------------------------------------------------------------------
                # DEFINE TRIANGLE COORDINATES
                # --------------------------------------------------------------------------------------------------------------
                v0 = np.array([0.2, 0.3], dtype=real)
                v1 = np.array([2.0, -0.5], dtype=real)
                v2 = np.array([2.0, 3.0], dtype=real)
                v3 = np.array([0.0, 3.1], dtype=real)
                vertices = np.array([v0, v1, v2, v3], dtype=real).T

                # --------------------------------------------------------------------------------------------------------------
                # BUILD CELL
                # --------------------------------------------------------------------------------------------------------------
                cell = Shape(ShapeType.QUADRANGLE, vertices)
                x_c = cell.centroid
                h_c = cell.diameter
                _io = finite_element.construction_integration_order
                cell_quadrature_points = cell.get_quadrature_points(_io)
                cell_quadrature_weights = cell.get_quadrature_weights(_io)
                cell_quadrature_size = cell.get_quadrature_size(_io)

                # --------------------------------------------------------------------------------------------------------------
                # PLOT CELL
                # --------------------------------------------------------------------------------------------------------------
                # --- PLOT VERTICES AND CENTROID
                plt.scatter(v0[0], v0[1], c="b")
                plt.scatter(v1[0], v1[1], c="b")
                plt.scatter(v2[0], v2[1], c="b")
                plt.scatter(v3[0], v3[1], c="b")
                plt.scatter(x_c[0], x_c[1], c="b")
                # --- PLOT QUADRATURE POINTS
                for _qc in range(cell_quadrature_size):
                    _x_qc = cell_quadrature_points[:, _qc]
                    plt.scatter(_x_qc[0], _x_qc[1], c="g")
                # --- PRINT QUADRATURE POINTS AND WEIGHTS
                for _qc in range(cell_quadrature_size):
                    _x_qc = cell_quadrature_points[:, _qc]
                    _w_qc = cell_quadrature_weights[_qc]
                    if verbose == True:
                        print("QUAD_POINT : {} | QUAD_WEIGHT : {}".format(_x_qc, _w_qc))
                # --- SET PLOT OPTIONS
                plt.gca().set_aspect("equal", adjustable="box")
                plt.grid()
                plt.show()

                # --------------------------------------------------------------------------------------------------------------
                # CHECK INTEGRATION IN CELL
                # --------------------------------------------------------------------------------------------------------------
                bases = [cell_basis_k, cell_basis_l]
                coefs = [coefficients_k, coefficients_l]
                # scheme = quadpy.c2.get_good_scheme(2 * finite_element.construction_integration_order)
                scheme = quadpy.c2.get_good_scheme(finite_element.construction_integration_order)
                for basis_0, coef_0 in zip(bases, coefs):
                    order_0 = basis_0.polynomial_order
                    for basis_1, coef_1 in zip(bases, coefs):
                        order_1 = basis_1.polynomial_order
                        for _i in range(euclidean_dimension):
                            for _j in range(euclidean_dimension):
                                mass_mat = np.zeros((basis_0.dimension, basis_1.dimension), dtype=real)
                                stif_mat = np.zeros((basis_0.dimension, basis_1.dimension), dtype=real)
                                advc_mat = np.zeros((basis_0.dimension, basis_1.dimension), dtype=real)
                                for _qc in range(cell_quadrature_size):
                                    _x_qc = cell_quadrature_points[:, _qc]
                                    _w_qc = cell_quadrature_weights[_qc]
                                    phi_0 = basis_0.evaluate_function(_x_qc, x_c, h_c)
                                    phi_1 = basis_1.evaluate_function(_x_qc, x_c, h_c)
                                    d_phi_0_i = basis_0.evaluate_derivative(_x_qc, x_c, h_c, _i)
                                    d_phi_1_j = basis_1.evaluate_derivative(_x_qc, x_c, h_c, _j)
                                    mass_mat += _w_qc * np.tensordot(phi_0, phi_1, axes=0)
                                    stif_mat += _w_qc * np.tensordot(d_phi_0_i, d_phi_1_j, axes=0)
                                    advc_mat += _w_qc * np.tensordot(phi_0, d_phi_1_j, axes=0)
                                mass_integral = coef_0 @ mass_mat @ coef_1
                                stif_integral = coef_0 @ stif_mat @ coef_1
                                advc_integral = coef_0 @ advc_mat @ coef_1
                                f_mass_check = lambda x: test_function(order_0, x, x_c, h_c, coef_0) * test_function(
                                    order_1, x, x_c, h_c, coef_1
                                )
                                f_stif_check = lambda x: test_function_derivative(
                                    order_0, x, x_c, h_c, _i, coef_0
                                ) * test_function_derivative(order_1, x, x_c, h_c, _j, coef_1)
                                f_advc_check = lambda x: test_function(
                                    order_0, x, x_c, h_c, coef_0
                                ) * test_function_derivative(order_1, x, x_c, h_c, _j, coef_1)
                                mass_integral_check = scheme.integrate(
                                    f_mass_check, [[vertices[:, 0], vertices[:, 1]], [vertices[:, 3], vertices[:, 2]]]
                                )
                                stif_integral_check = scheme.integrate(
                                    f_stif_check, [[vertices[:, 0], vertices[:, 1]], [vertices[:, 3], vertices[:, 2]]]
                                )
                                advc_integral_check = scheme.integrate(
                                    f_advc_check, [[vertices[:, 0], vertices[:, 1]], [vertices[:, 3], vertices[:, 2]]]
                                )
                                rtol = 1.0e-15
                                atol = 1.0e-15
                                print('MASS INTEGRAL CHECK | ORDER : {} | ELEM : {}'.format(polynomial_order, element_type))
                                print("- QUADPY : {}".format(mass_integral_check))
                                print("- H2O : {}".format(mass_integral))
                                np.testing.assert_allclose(mass_integral_check, mass_integral, rtol=rtol, atol=atol)
                                print('STIFFNESS INTEGRAL CHECK | ORDER : {} | ELEM : {}'.format(polynomial_order, element_type))
                                print("- QUADPY : {}".format(stif_integral_check))
                                print("- H2O : {}".format(stif_integral))
                                np.testing.assert_allclose(stif_integral_check, stif_integral, rtol=rtol, atol=atol)
                                print('ADVECTION INTEGRAL CHECK | ORDER : {} | ELEM : {}'.format(polynomial_order, element_type))
                                print("- QUADPY : {}".format(advc_integral_check))
                                print("- H2O : {}".format(advc_integral))
                                np.testing.assert_allclose(advc_integral_check, advc_integral, rtol=rtol, atol=atol)
