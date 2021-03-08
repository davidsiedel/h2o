from random import uniform
from unittest import TestCase

import matplotlib.pyplot as plt
import quadpy

from h2o.fem.basis.bases.monomial import Monomial
from h2o.geometry.shape import Shape, get_rotation_matrix
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
import h2o.fem.element.operators.operator_gradient as gradop
import h2o.fem.element.operators.operator_stabilization as stabop
from h2o.h2o import *

np.set_printoptions(precision=16)
np.set_printoptions(linewidth=1)


class TestElementTriangle(TestCase):
    def test_face_segment(self):

        euclidean_dimension = 2
        polynomial_orders = [1, 2, 3]
        element_types = [ElementType.HDG_LOW, ElementType.HDG_EQUAL, ElementType.HDG_HIGH]
        for polynomial_order in polynomial_orders:
            for element_type in element_types:
                hyperplane_dimension = euclidean_dimension - 1
                # --------------------------------------------------------------------------------------------------------------
                # DEFINE POLYNOMIAL ORDER AND INTEGRATION ORDER
                # --------------------------------------------------------------------------------------------------------------
                finite_element = FiniteElement(
                    element_type=element_type,
                    polynomial_order=polynomial_order,
                    euclidean_dimension=euclidean_dimension,
                )
                face_basis_k = finite_element.face_basis_k
                # --------------------------------------------------------------------------------------------------------------
                # DEFINE RANDOM POLYNOMIAL COEFFICIENTS
                # --------------------------------------------------------------------------------------------------------------
                amplitude = 1.0
                range_min = -1.0 * amplitude
                range_max = +1.0 * amplitude
                coefficients = np.array([uniform(range_min, range_max) for _i in range(face_basis_k.dimension)])
                print("COEFS : \n{}".format(coefficients))

                # --------------------------------------------------------------------------------------------------------------
                # DEFINE MONOMIAL VALUES COMPUTATION
                # --------------------------------------------------------------------------------------------------------------
                def test_function(point: ndarray, centroid: ndarray, diameter: float, coefficients: ndarray) -> float:
                    basis = Monomial(polynomial_order, hyperplane_dimension)
                    value = 0.0
                    for _i, _exponent in enumerate(basis.exponents):
                        prod = 1.0
                        for _x_dir in range(basis.exponents.shape[1]):
                            # prod *= coefficients[_i] * (((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir])
                            prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
                        prod *= coefficients[_i]
                        value += prod
                    return value

                def test_function_derivative(
                    point: ndarray, centroid: ndarray, diameter: float, direction: int, coefficients: ndarray
                ) -> float:
                    basis = Monomial(polynomial_order, hyperplane_dimension)
                    value = 0.0
                    for _i, _exponent in enumerate(basis.exponents):
                        prod = 1.0
                        for _x_dir in range(basis.exponents.shape[1]):
                            if _x_dir == direction:
                                prod *= (_exponent[_x_dir] / diameter) * (
                                    ((point[_x_dir] - centroid[_x_dir]) / diameter) ** (_exponent[_x_dir] - 1)
                                )
                            else:
                                prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
                        prod *= coefficients[_i]
                        value += prod
                    return value

                # def get_phi_vector(point: ndarray, centroid: ndarray, diameter: float) -> ndarray:
                #     phi_vector = np.zeros((face_basis_k.dimension,), dtype=real)
                #     for _i, _exponent in enumerate(face_basis_k.exponents):
                #         prod = 1.0
                #         for _x_dir in range(face_basis_k.exponents.shape[1]):
                #             prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
                #         phi_vector[_i] += prod
                #     return phi_vector

                # --------------------------------------------------------------------------------------------------------------
                # DEFINE SEGMENT COORDINATES
                # --------------------------------------------------------------------------------------------------------------
                v_0 = np.array([1.0, 1.0], dtype=real)
                v_1 = np.array([2.0, 5.0], dtype=real)
                segment_vertices = np.array([v_0, v_1], dtype=real).T

                # --------------------------------------------------------------------------------------------------------------
                # BUILD FACE
                # --------------------------------------------------------------------------------------------------------------
                face_segment = Shape(ShapeType.SEGMENT, segment_vertices)
                face_segment_rotation_matrix = get_rotation_matrix(face_segment.type, face_segment.vertices)
                x_f = face_segment.centroid
                h_f = face_segment.diameter
                # --- PROJECT ON HYPERPLANE
                s_f = (face_segment_rotation_matrix @ face_segment.centroid)[:-1]
                s_0 = (face_segment_rotation_matrix @ v_0)[:-1]
                s_1 = (face_segment_rotation_matrix @ v_1)[:-1]
                _io = finite_element.construction_integration_order
                face_segment_quadrature_points = face_segment.get_quadrature_points(_io)
                face_segment_quadrature_weights = face_segment.get_quadrature_weights(_io)
                face_segment_quadrature_size = face_segment.get_quadrature_size(_io)

                # --------------------------------------------------------------------------------------------------------------
                # PLOT FACE
                # --------------------------------------------------------------------------------------------------------------
                # --- PLOT VERTICES AND CENTROID
                plt.scatter(v_0[0], v_0[1], c="b")
                plt.scatter(v_1[0], v_1[1], c="b")
                plt.scatter(x_f[0], x_f[1], c="b")
                # --- PLOT PROJECTED VERTICES AND CENTROID
                plt.scatter(s_0, 0.0, c="black")
                plt.scatter(s_1, 0.0, c="black")
                plt.scatter(s_f, 0.0, c="black")
                # --- PRINT QUADRATURE POINTS AND WEIGHTS
                for _qf in range(face_segment_quadrature_size):
                    x_qf = face_segment_quadrature_points[:, _qf]
                    # --- PLOT QUADRATURE POINT
                    plt.scatter(x_qf[0], x_qf[1], c="g")
                    # --- PLOT PROJECTED QUADRATURE POINT
                    s_qf = (face_segment_rotation_matrix @ x_qf)[:-1]
                    plt.scatter(s_qf, 0.0, c="grey")
                # --- PRINT QUADRATURE POINTS AND WEIGHTS
                for _qf in range(face_segment_quadrature_size):
                    x_qf = face_segment_quadrature_points[:, _qf]
                    w_qf = face_segment_quadrature_weights[_qf]
                    print("QUAD_POINT : {} | QUAD_WEIGHT : {}".format(x_qf, w_qf))
                # --- SET PLOT OPTIONS
                plt.gca().set_aspect("equal", adjustable="box")
                plt.grid()
                plt.show()

                # --------------------------------------------------------------------------------------------------------------
                # CHECK DISTANCES
                # --------------------------------------------------------------------------------------------------------------
                print("DIAMETER : {}".format(h_f))
                print("DIST ORIGINAL : {}".format(np.linalg.norm(v_0 - v_1)))
                print("DIST PROJECTION : {}".format(np.linalg.norm(s_0 - s_1)))

                # --------------------------------------------------------------------------------------------------------------
                # CHECK INTEGRATION IN FACE
                # --------------------------------------------------------------------------------------------------------------
                scheme = quadpy.c1.gauss_legendre(2 * _io)
                for _i in range(hyperplane_dimension):
                    for _j in range(hyperplane_dimension):
                        mass_mat = np.zeros((face_basis_k.dimension, face_basis_k.dimension), dtype=real)
                        stif_mat = np.zeros((face_basis_k.dimension, face_basis_k.dimension), dtype=real)
                        advc_mat = np.zeros((face_basis_k.dimension, face_basis_k.dimension), dtype=real)
                        for _qf in range(face_segment_quadrature_size):
                            _x_qf = face_segment_quadrature_points[:, _qf]
                            _s_qf = (face_segment_rotation_matrix @ _x_qf)[:-1]
                            _w_qf = face_segment_quadrature_weights[_qf]
                            phi_0 = face_basis_k.evaluate_function(_s_qf, s_f, h_f)
                            d_phi_0_i = face_basis_k.evaluate_derivative(_s_qf, s_f, h_f, _i)
                            d_phi_0_j = face_basis_k.evaluate_derivative(_s_qf, s_f, h_f, _j)
                            mass_mat += _w_qf * np.tensordot(phi_0, phi_0, axes=0)
                            stif_mat += _w_qf * np.tensordot(d_phi_0_i, d_phi_0_j, axes=0)
                            advc_mat += _w_qf * np.tensordot(phi_0, d_phi_0_i, axes=0)
                        mass_integral = coefficients @ mass_mat @ coefficients
                        stif_integral = coefficients @ stif_mat @ coefficients
                        advc_integral = coefficients @ advc_mat @ coefficients
                        f_mass_check = lambda x: test_function(np.array([x]), s_f, h_f, coefficients) * test_function(
                            np.array([x]), s_f, h_f, coefficients
                        )
                        f_stif_check = lambda x: test_function_derivative(
                            np.array([x]), s_f, h_f, _i, coefficients
                        ) * test_function_derivative(np.array([x]), s_f, h_f, _j, coefficients)
                        f_advc_check = lambda x: test_function(
                            np.array([x]), s_f, h_f, coefficients
                        ) * test_function_derivative(np.array([x]), s_f, h_f, _j, coefficients)
                        p_0 = s_0[0]
                        p_1 = s_1[0]
                        mass_integral_check = scheme.integrate(f_mass_check, [p_0, p_1])
                        stif_integral_check = scheme.integrate(f_stif_check, [p_0, p_1])
                        advc_integral_check = scheme.integrate(f_advc_check, [p_0, p_1])
                        rtol = 1.0e-15
                        atol = 1.0e-15
                        np.testing.assert_allclose(mass_integral_check, mass_integral, rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stif_integral_check, stif_integral, rtol=rtol, atol=atol)
                        np.testing.assert_allclose(advc_integral_check, advc_integral, rtol=rtol, atol=atol)

    def test_cell_triangle(self):

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
                v0 = np.array([1.0, 1.7], dtype=real)
                v1 = np.array([2.0, 1.6], dtype=real)
                v2 = np.array([1.9, 3.0], dtype=real)
                triangle_vertices = np.array([v0, v1, v2], dtype=real).T

                # --------------------------------------------------------------------------------------------------------------
                # BUILD CELL
                # --------------------------------------------------------------------------------------------------------------
                cell_triangle = Shape(ShapeType.TRIANGLE, triangle_vertices)
                x_c = cell_triangle.centroid
                h_c = cell_triangle.diameter
                _io = finite_element.construction_integration_order
                cell_quadrature_points = cell_triangle.get_quadrature_points(_io)
                cell_quadrature_weights = cell_triangle.get_quadrature_weights(_io)
                cell_quadrature_size = cell_triangle.get_quadrature_size(_io)

                # --------------------------------------------------------------------------------------------------------------
                # PLOT CELL
                # --------------------------------------------------------------------------------------------------------------
                # --- PLOT VERTICES AND CENTROID
                plt.scatter(v0[0], v0[1], c="b")
                plt.scatter(v1[0], v1[1], c="b")
                plt.scatter(v2[0], v2[1], c="b")
                plt.scatter(x_c[0], x_c[1], c="b")
                # --- PLOT QUADRATURE POINTS
                for _qc in range(cell_quadrature_size):
                    _x_qc = cell_quadrature_points[:, _qc]
                    plt.scatter(_x_qc[0], _x_qc[1], c="g")
                # --- PRINT QUADRATURE POINTS AND WEIGHTS
                for _qc in range(cell_quadrature_size):
                    _x_qc = cell_quadrature_points[:, _qc]
                    _w_qc = cell_quadrature_weights[_qc]
                    print("QUAD_POINT : {} | QUAD_WEIGHT : {}".format(_x_qc, _w_qc))
                # --- SET PLOT OPTIONS
                plt.gca().set_aspect("equal", adjustable="box")
                plt.grid()
                plt.show()

                # --------------------------------------------------------------------------------------------------------------
                # CHECK INTEGRATION IN CELL
                # --------------------------------------------------------------------------------------------------------------
                bases = [cell_basis_k, cell_basis_l]
                # orders = [face_polynomial_order, cell_polynomial_order]
                coefs = [coefficients_k, coefficients_l]
                scheme = quadpy.t2.get_good_scheme(2 * finite_element.construction_integration_order)
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
                                mass_integral_check = scheme.integrate(f_mass_check, triangle_vertices.T)
                                stif_integral_check = scheme.integrate(f_stif_check, triangle_vertices.T)
                                advc_integral_check = scheme.integrate(f_advc_check, triangle_vertices.T)
                                rtol = 1.0e-15
                                atol = 1.0e-15
                                np.testing.assert_allclose(mass_integral_check, mass_integral, rtol=rtol, atol=atol)
                                np.testing.assert_allclose(stif_integral_check, stif_integral, rtol=rtol, atol=atol)
                                np.testing.assert_allclose(advc_integral_check, advc_integral, rtol=rtol, atol=atol)

        # --------------------------------------------------------------------------------------------------------------
        # DEFINE POLYNOMIAL ORDER AND INTEGRATION ORDER
        # --------------------------------------------------------------------------------------------------------------
        # face_polynomial_order = 2
        # cell_polynomial_order = face_polynomial_order + 1
        # euclidean_dimension = 2
        # integration_order = 2 * (face_polynomial_order + 1)
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # DEFINE POLYNOMIAL BASIS
        # # --------------------------------------------------------------------------------------------------------------
        # cell_basis_k = Monomial(face_polynomial_order, euclidean_dimension)
        # cell_basis_l = Monomial(cell_polynomial_order, euclidean_dimension)
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # DEFINE RANDOM POLYNOMIAL COEFFICIENTS
        # # --------------------------------------------------------------------------------------------------------------
        # range_min = -3.0
        # range_max = +3.0
        # coefficients_k = np.array([uniform(range_min, range_max) for _i in range(cell_basis_k.dimension)])
        # coefficients_l = np.array([uniform(range_min, range_max) for _i in range(cell_basis_l.dimension)])
        # print("COEFS_K : \n{}".format(coefficients_k))
        # print("COEFS_L : \n{}".format(coefficients_l))
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # DEFINE MONOMIAL VALUES COMPUTATION
        # # --------------------------------------------------------------------------------------------------------------
        # def test_function(
        #     basis: Monomial, point: ndarray, centroid: ndarray, diameter: float, coefficients: ndarray
        # ) -> float:
        #     value = 0.0
        #     for _i, _exponent in enumerate(basis.exponents):
        #         prod = 1.0
        #         for _x_dir in range(basis.exponents.shape[1]):
        #             prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
        #         prod *= coefficients[_i]
        #         value += prod
        #     return value
        #
        # def test_function_derivative(
        #     basis: Monomial, point: ndarray, centroid: ndarray, diameter: float, direction: int, coefficients: ndarray
        # ) -> float:
        #     value = 0.0
        #     for _i, _exponent in enumerate(basis.exponents):
        #         prod = 1.0
        #         for _x_dir in range(basis.exponents.shape[1]):
        #             if _x_dir == direction:
        #                 prod *= (_exponent[_x_dir] / diameter) * (
        #                     ((point[_x_dir] - centroid[_x_dir]) / diameter) ** (_exponent[_x_dir] - 1)
        #                 )
        #             else:
        #                 prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
        #         prod *= coefficients[_i]
        #         value += prod
        #     return value
        #
        # def get_phi_vector(point: ndarray, centroid: ndarray, diameter: float) -> ndarray:
        #     phi_vector = np.zeros((cell_basis_k.dimension,), dtype=real)
        #     for _i, _exponent in enumerate(cell_basis_k.exponents):
        #         prod = 1.0
        #         for _x_dir in range(cell_basis_k.exponents.shape[1]):
        #             prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
        #         phi_vector[_i] += prod
        #     return phi_vector
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # DEFINE TRIANGLE COORDINATES
        # # --------------------------------------------------------------------------------------------------------------
        # v0 = np.array([1.0, 1.7], dtype=real)
        # v1 = np.array([2.0, 1.6], dtype=real)
        # v2 = np.array([1.9, 3.0], dtype=real)
        # triangle_vertices = np.array([v0, v1, v2], dtype=real).T
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # BUILD CELL
        # # --------------------------------------------------------------------------------------------------------------
        # cell_triangle = Cell(ShapeType.TRIANGLE, triangle_vertices, integration_order)
        # x_c = cell_triangle.shape.centroid
        # h_c = cell_triangle.shape.diameter
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # PLOT CELL
        # # --------------------------------------------------------------------------------------------------------------
        # # --- PLOT VERTICES AND CENTROID
        # plt.scatter(v0[0], v0[1], c="b")
        # plt.scatter(v1[0], v1[1], c="b")
        # plt.scatter(v2[0], v2[1], c="b")
        # plt.scatter(x_c[0], x_c[1], c="b")
        # # --- PLOT QUADRATURE POINTS
        # for _qc in range(len(cell_triangle.quadrature_weights)):
        #     _x_qc = cell_triangle.quadrature_points[:, _qc]
        #     plt.scatter(_x_qc[0], _x_qc[1], c="g")
        # # --- PRINT QUADRATURE POINTS AND WEIGHTS
        # for _qc in range(len(cell_triangle.quadrature_weights)):
        #     _x_qc = cell_triangle.quadrature_points[:, _qc]
        #     _w_qc = cell_triangle.quadrature_weights[_qc]
        #     print("QUAD_POINT : {} | QUAD_WEIGHT : {}".format(_x_qc, _w_qc))
        # # --- SET PLOT OPTIONS
        # plt.gca().set_aspect("equal", adjustable="box")
        # plt.grid()
        # plt.show()
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # CHECK INTEGRATION IN CELL
        # # --------------------------------------------------------------------------------------------------------------
        # bases = [cell_basis_k, cell_basis_l]
        # orders = [face_polynomial_order, cell_polynomial_order]
        # coefs = [coefficients_k, coefficients_l]
        # scheme = quadpy.t2.get_good_scheme(2 * integration_order)
        # for basis_0, order_0, coef_0 in zip(bases, orders, coefs):
        #     for basis_1, order_1, coef_1 in zip(bases, orders, coefs):
        #         for _i in range(euclidean_dimension):
        #             for _j in range(euclidean_dimension):
        #                 mass_mat = np.zeros((basis_0.dimension, basis_1.dimension), dtype=real)
        #                 stif_mat = np.zeros((basis_0.dimension, basis_1.dimension), dtype=real)
        #                 advc_mat = np.zeros((basis_0.dimension, basis_1.dimension), dtype=real)
        #                 for _qc in range(len(cell_triangle.quadrature_weights)):
        #                     _x_qc = cell_triangle.quadrature_points[:, _qc]
        #                     _w_qc = cell_triangle.quadrature_weights[_qc]
        #                     phi_0 = basis_0.get_phi_vector(_x_qc, x_c, h_c)
        #                     phi_1 = basis_1.get_phi_vector(_x_qc, x_c, h_c)
        #                     d_phi_0_i = basis_0.get_d_phi_vector(_x_qc, x_c, h_c, _i)
        #                     d_phi_1_j = basis_1.get_d_phi_vector(_x_qc, x_c, h_c, _j)
        #                     mass_mat += _w_qc * np.tensordot(phi_0, phi_1, axes=0)
        #                     stif_mat += _w_qc * np.tensordot(d_phi_0_i, d_phi_1_j, axes=0)
        #                     advc_mat += _w_qc * np.tensordot(phi_0, d_phi_1_j, axes=0)
        #                 mass_integral = coef_0 @ mass_mat @ coef_1
        #                 stif_integral = coef_0 @ stif_mat @ coef_1
        #                 advc_integral = coef_0 @ advc_mat @ coef_1
        #                 f_mass_check = lambda x: test_function(basis_0, x, x_c, h_c, coef_0) * test_function(
        #                     basis_1, x, x_c, h_c, coef_1
        #                 )
        #                 f_stif_check = lambda x: test_function_derivative(
        #                     basis_0, x, x_c, h_c, _i, coef_0
        #                 ) * test_function_derivative(basis_1, x, x_c, h_c, _j, coef_1)
        #                 f_advc_check = lambda x: test_function(basis_0, x, x_c, h_c, coef_0) * test_function_derivative(
        #                     basis_1, x, x_c, h_c, _j, coef_1
        #                 )
        #                 mass_integral_check = scheme.integrate(f_mass_check, triangle_vertices.T)
        #                 stif_integral_check = scheme.integrate(f_stif_check, triangle_vertices.T)
        #                 advc_integral_check = scheme.integrate(f_advc_check, triangle_vertices.T)
        #                 rtol = 1.0e-15
        #                 atol = 1.0e-15
        #                 np.testing.assert_allclose(mass_integral_check, mass_integral, rtol=rtol, atol=atol)
        #                 np.testing.assert_allclose(stif_integral_check, stif_integral, rtol=rtol, atol=atol)
        #                 np.testing.assert_allclose(advc_integral_check, advc_integral, rtol=rtol, atol=atol)
        return

    def test_element_triangle(self):
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

                field = Field("TEST", FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN)

                # --------------------------------------------------------------------------------------------------------------
                # DEFINE POLYNOMIAL BASIS
                # --------------------------------------------------------------------------------------------------------------
                cell_basis_k = finite_element.cell_basis_k
                cell_basis_l = finite_element.cell_basis_l
                face_basis_k = finite_element.face_basis_k

                # --------------------------------------------------------------------------------------------------------------
                # DEFINE RANDOM POLYNOMIAL COEFFICIENTS
                # --------------------------------------------------------------------------------------------------------------
                range_min = -3.0
                range_max = +3.0
                coefficients_k_list = []
                coefficients_l_list = []
                for _i in range(field.field_dimension):
                    coefficients_k = np.array(
                        [uniform(range_min, range_max) for _iloc in range(cell_basis_k.dimension)]
                    )
                    coefficients_l = np.array(
                        [uniform(range_min, range_max) for _iloc in range(cell_basis_l.dimension)]
                    )
                    coefficients_k_list.append(coefficients_k)
                    coefficients_l_list.append(coefficients_l)
                # print("COEFS_K : \n{}".format(coefficients_k))
                # print("COEFS_L : \n{}".format(coefficients_l))

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

                # --------------------------------------------------------------------------------------------------------------
                # DEFINE TRIANGLE COORDINATES
                # --------------------------------------------------------------------------------------------------------------
                v0 = np.array([1.0, 1.7], dtype=real)
                v1 = np.array([2.0, 1.6], dtype=real)
                v2 = np.array([1.9, 3.0], dtype=real)
                triangle_vertices = np.array([v0, v1, v2], dtype=real).T

                # --------------------------------------------------------------------------------------------------------------
                # BUILD CELL
                # --------------------------------------------------------------------------------------------------------------
                cell_triangle = Shape(ShapeType.TRIANGLE, triangle_vertices)

                # --------------------------------------------------------------------------------------------------------------
                # BUILD FACES
                # --------------------------------------------------------------------------------------------------------------
                faces_segment = [
                    Shape(ShapeType.SEGMENT, triangle_vertices[:, [0, 1]]),
                    Shape(ShapeType.SEGMENT, triangle_vertices[:, [1, 2]]),
                    Shape(ShapeType.SEGMENT, triangle_vertices[:, [2, 0]]),
                ]

                def get_element_projection_vector(cell: Shape, faces: List[Shape], function: List[Callable]):
                    _d = euclidean_dimension
                    _dx = field.field_dimension
                    _cl = cell_basis_l.dimension
                    _fk = face_basis_k.dimension
                    _nf = len(faces)
                    _es = _dx * (_cl + _nf * _fk)
                    #
                    x_c = cell.centroid
                    h_c = cell.diameter
                    _io = finite_element.construction_integration_order
                    cell_quadrature_points = cell.get_quadrature_points(_io)
                    cell_quadrature_weights = cell.get_quadrature_weights(_io)
                    cell_quadrature_size = cell.get_quadrature_size(_io)
                    matrix = np.zeros((_es, _es), dtype=real)
                    vector = np.zeros((_es,), dtype=real)
                    for _dir in range(_dx):
                        m_mas = np.zeros((_cl, _cl), dtype=real)
                        vc = np.zeros((_cl,), dtype=real)
                        for qc in range(cell_quadrature_size):
                            x_q_c = cell_quadrature_points[:, qc]
                            w_q_c = cell_quadrature_weights[qc]
                            phi_l = cell_basis_l.evaluate_function(x_q_c, x_c, h_c)
                            m_mas += w_q_c * np.tensordot(phi_l, phi_l, axes=0)
                            vc += w_q_c * phi_l * function[_dir](x_q_c)
                        _i = _cl * _dir
                        _j = _cl * (_dir + 1)
                        matrix[_i:_j, _i:_j] += m_mas
                        vector[_i:_j] += vc
                    for _f, face in enumerate(faces):
                        face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
                        x_f = face.centroid
                        h_f = face.diameter
                        # --- PROJECT ON HYPERPLANE
                        s_f = (face_rotation_matrix @ x_f)[:-1]
                        _io = finite_element.construction_integration_order
                        face_quadrature_points = face.get_quadrature_points(_io)
                        face_quadrature_weights = face.get_quadrature_weights(_io)
                        face_quadrature_size = face.get_quadrature_size(_io)
                        for _dir in range(_dx):
                            m_mas_f = np.zeros((_fk, _fk), dtype=real)
                            vf = np.zeros((_fk,), dtype=real)
                            for qf in range(face_quadrature_size):
                                x_q_f = face_quadrature_points[:, qf]
                                w_q_f = face_quadrature_weights[qf]
                                # s_f = (face_rotation_matrix @ x_f)[:-1]
                                s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
                                psi_k = face_basis_k.evaluate_function(s_q_f, s_f, h_f)
                                m_mas_f += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
                                vf += w_q_f * psi_k * function[_dir](x_q_f)
                            _i = _cl * _dx + _f * _fk * _dx + _dir * _fk
                            _j = _cl * _dx + _f * _fk * _dx + (_dir + 1) * _fk
                            matrix[_i:_j, _i:_j] += m_mas_f
                            vector[_i:_j] += vf
                    projection_vector = np.linalg.solve(matrix, vector)
                    return projection_vector

                def get_gradient_projection_vector(cell: Shape, function: Callable):
                    _d = euclidean_dimension
                    _dx = field.field_dimension
                    _ck = cell_basis_k.dimension
                    #
                    x_c = cell.centroid
                    h_c = cell.diameter
                    _io = finite_element.construction_integration_order
                    cell_quadrature_points = cell.get_quadrature_points(_io)
                    cell_quadrature_weights = cell.get_quadrature_weights(_io)
                    cell_quadrature_size = cell.get_quadrature_size(_io)
                    matrix = np.zeros((_ck, _ck), dtype=real)
                    vector = np.zeros((_ck,), dtype=real)
                    for qc in range(cell_quadrature_size):
                        x_q_c = cell_quadrature_points[:, qc]
                        w_q_c = cell_quadrature_weights[qc]
                        phi_k = cell_basis_k.evaluate_function(x_q_c, x_c, h_c)
                        matrix += w_q_c * np.tensordot(phi_k, phi_k, axes=0)
                        vector += w_q_c * phi_k * function(x_q_c)
                    projection_vector = np.linalg.solve(matrix, vector)
                    return projection_vector

                # fun = [
                #     lambda x : test_function(finite_element.cell_basis_l.polynomial_order, x, cell_triangle.centroid, cell_triangle.diameter, coefficients_l_list[_i])
                # ]

                fun = [
                    lambda x: test_function(
                        finite_element.cell_basis_l.polynomial_order,
                        x,
                        cell_triangle.centroid,
                        cell_triangle.diameter,
                        coefficients_l_list[0],
                    ),
                    lambda x: test_function(
                        finite_element.cell_basis_l.polynomial_order,
                        x,
                        cell_triangle.centroid,
                        cell_triangle.diameter,
                        coefficients_l_list[1],
                    )
                    # for _iici in range(field.field_dimension)
                ]

                # fun_grad_regular = [
                #     [
                #         lambda x: test_function_derivative(
                #             finite_element.cell_basis_l.polynomial_order,
                #             x,
                #             cell_triangle.centroid,
                #             cell_triangle.diameter,
                #             _j,
                #             coefficients_l_list[_i],
                #         )
                #         for _j in range(field.field_dimension)
                #     ]
                #     for _i in range(field.field_dimension)
                # ]
                fun_grad_regular = [
                    [
                        lambda x: test_function_derivative(
                            finite_element.cell_basis_l.polynomial_order,
                            x,
                            cell_triangle.centroid,
                            cell_triangle.diameter,
                            0,
                            coefficients_l_list[0],
                        ),
                        lambda x: test_function_derivative(
                            finite_element.cell_basis_l.polynomial_order,
                            x,
                            cell_triangle.centroid,
                            cell_triangle.diameter,
                            1,
                            coefficients_l_list[0],
                        ),
                    ],
                    [
                        lambda x: test_function_derivative(
                            finite_element.cell_basis_l.polynomial_order,
                            x,
                            cell_triangle.centroid,
                            cell_triangle.diameter,
                            0,
                            coefficients_l_list[1],
                        ),
                        lambda x: test_function_derivative(
                            finite_element.cell_basis_l.polynomial_order,
                            x,
                            cell_triangle.centroid,
                            cell_triangle.diameter,
                            1,
                            coefficients_l_list[1],
                        ),
                    ],
                ]

                # fun_grad_symmetric = [
                #     [
                #         lambda x: (1./2.) *
                #         test_function_derivative(
                #             finite_element.cell_basis_l.polynomial_order,
                #             x,
                #             cell_triangle.centroid,
                #             cell_triangle.diameter,
                #             _j,
                #             coefficients_l_list[_i],
                #         ) + test_function_derivative(
                #             finite_element.cell_basis_l.polynomial_order,
                #             x,
                #             cell_triangle.centroid,
                #             cell_triangle.diameter,
                #             _i,
                #             coefficients_l_list[_j],
                #         )
                #         for _j in range(field.field_dimension)
                #     ]
                #     for _i in range(field.field_dimension)
                # ]
                fun_grad_symmetric = [
                    [
                        lambda x: (1.0 / 2.0)
                        * (
                            test_function_derivative(
                                finite_element.cell_basis_l.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                0,
                                coefficients_l_list[0],
                            )
                            + test_function_derivative(
                                finite_element.cell_basis_l.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                0,
                                coefficients_l_list[0],
                            )
                        ),
                        lambda x: (1.0 / 2.0)
                        * (
                            test_function_derivative(
                                finite_element.cell_basis_l.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                1,
                                coefficients_l_list[0],
                            )
                            + test_function_derivative(
                                finite_element.cell_basis_l.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                0,
                                coefficients_l_list[1],
                            )
                        ),
                    ],
                    [
                        lambda x: (1.0 / 2.0)
                        * (
                            test_function_derivative(
                                finite_element.cell_basis_l.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                0,
                                coefficients_l_list[1],
                            )
                            + test_function_derivative(
                                finite_element.cell_basis_l.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                1,
                                coefficients_l_list[0],
                            )
                        ),
                        lambda x: (1.0 / 2.0)
                        * (
                            test_function_derivative(
                                finite_element.cell_basis_l.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                1,
                                coefficients_l_list[1],
                            )
                            + test_function_derivative(
                                finite_element.cell_basis_l.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                1,
                                coefficients_l_list[1],
                            )
                        ),
                    ],
                ]

                # fun = [lambda x: np.cos(x[0]) * np.sin(x[1]), lambda x: x[0] * x[1]]
                #
                # fun_grad_sym = [
                #     lambda x: -(np.sin(x[0]) * np.sin(x[1])),
                #     lambda x: x[0],
                #     lambda x: (1.0 / 2.0) * (np.cos(x[0]) * np.cos(x[1]) + x[1]),
                #     lambda x: (1.0 / 2.0) * (np.cos(x[0]) * np.cos(x[1]) + x[1]),
                # ]
                #
                # fun_grad_full = [
                #     lambda x: -(np.sin(x[0]) * np.sin(x[1])),
                #     lambda x: x[0],
                #     lambda x: (np.cos(x[0]) * np.cos(x[1])),
                #     lambda x: x[1],
                # ]
                #
                # fun = [lambda x: 2.0 * x[0] + 3.0 * x[1], lambda x: 6.0 * x[0] * x[1]]
                #
                # fun_grad_sym = [
                #     lambda x: 2.0,
                #     lambda x: 6.0 * x[0],
                #     lambda x: (1.0 / 2.0) * (3.0 + 6.0 * x[1]),
                #     lambda x: (1.0 / 2.0) * (3.0 + 6.0 * x[1]),
                # ]
                #
                # fun_grad_full = [lambda x: 2.0, lambda x: 6.0 * x[0], lambda x: 3.0, lambda x: 6.0 * x[1]]

                # --------------------------------------------------------------------------------------------------------------
                # BUILD AND TEST STABILIZATION
                # --------------------------------------------------------------------------------------------------------------

                fun_proj = get_element_projection_vector(cell_triangle, faces_segment, fun)
                # stab_matrix, stab_matrix_0, stab_matrix2 = get_stabilization_operator(cell_triangle, faces_segment)
                stabilization_operator = stabop.get_stabilization_operator2(
                    field, finite_element, cell_triangle, faces_segment
                )
                print(
                    "--- FUN PROJ | k : {} | l : {}".format(
                        cell_basis_k.polynomial_order, cell_basis_l.polynomial_order
                    )
                )
                print(fun_proj)
                print(
                    "--- STABILIZATION | k : {} | l : {}".format(
                        cell_basis_k.polynomial_order, cell_basis_l.polynomial_order
                    )
                )
                stab_vector = fun_proj @ stabilization_operator @ fun_proj
                print(stab_vector)

                correspondance = {0: (0, 0), 1: (1, 1), 2: (0, 1), 3: (1, 0)}
                for key, val in correspondance.items():
                    dir_x = val[0]
                    dir_y = val[1]
                    # rtol = 1.0e-12
                    # rtol = 1.0e-3
                    rtol = 1000000.
                    atol = 1.0e-11
                    print(
                        "--- SYMMETRIC GRADIENT | k : {} | l : {}".format(
                            cell_basis_k.polynomial_order, cell_basis_l.polynomial_order
                        )
                    )
                    fun_proj = get_element_projection_vector(cell_triangle, faces_segment, fun)
                    grad_comp = gradop.get_symmetric_gradient_component_matrix(
                        field, finite_element, cell_triangle, faces_segment, dir_x, dir_y
                    )
                    # fun_grad_proj = get_gradient_projection_vector(cell_triangle, fun_grad_sym[choice])
                    # fun_grad_proj = get_gradient_projection_vector(cell_triangle, fun_grad_symmetric[dir_x][dir_y])
                    fun_grad_proj = get_gradient_projection_vector(cell_triangle, fun_grad_symmetric[dir_y][dir_x])
                    grad_check = grad_comp @ fun_proj
                    print("- GRAD REC | {} | {}".format(dir_x, dir_y))
                    print(grad_check)
                    print("- GRAD PROJ | {} | {}".format(dir_x, dir_y))
                    print(fun_grad_proj)
                    np.testing.assert_allclose(grad_check, fun_grad_proj, rtol=rtol, atol=atol)
                    print(
                        "--- REGULAR GRADIENT | k : {} | l : {}".format(
                            cell_basis_k.polynomial_order, cell_basis_l.polynomial_order
                        )
                    )
                    fun_proj = get_element_projection_vector(cell_triangle, faces_segment, fun)
                    grad_comp = gradop.get_regular_gradient_component_matrix(
                        field, finite_element, cell_triangle, faces_segment, dir_x, dir_y
                    )
                    # fun_grad_proj = get_gradient_projection_vector(cell_triangle, fun_grad_sym[choice])
                    fun_grad_proj = get_gradient_projection_vector(cell_triangle, fun_grad_regular[dir_x][dir_y])
                    grad_check = grad_comp @ fun_proj
                    print("- GRAD REC | {} | {}".format(dir_x, dir_y))
                    print(grad_check)
                    print("- GRAD PROJ | {} | {}".format(dir_x, dir_y))
                    print(fun_grad_proj)
                    np.testing.assert_allclose(grad_check, fun_grad_proj, rtol=rtol, atol=atol)
                # choice = 1
                # dir_x = correspondance[choice][0]
                # dir_y = correspondance[choice][1]
                # rtol = 1.0e-3
                # atol = 1.0e-3
                # print("--- SYMMETRIC GRADIENT")
                # fun_proj = get_element_projection_vector(cell_triangle, faces_segment, fun)
                # grad_comp = gradop.get_symmetric_gradient_component_matrix(
                #     field, finite_element, cell_triangle, faces_segment, dir_x, dir_y
                # )
                # # fun_grad_proj = get_gradient_projection_vector(cell_triangle, fun_grad_sym[choice])
                # fun_grad_proj = get_gradient_projection_vector(cell_triangle, fun_grad_regular[dir_x][dir_y])
                # grad_check = grad_comp @ fun_proj
                # np.testing.assert_allclose(grad_check, fun_grad_proj, rtol=rtol, atol=atol)
                # # x_c = cell_triangle.shape.centroid
                # # h_c = cell_triangle.shape.diameter
                # # grad_check_val = 0.0
                # # fun_grad_proj_val = 0.0
                # # for qc in range(len(cell_triangle.quadrature_weights)):
                # #     w_qc = cell_triangle.quadrature_weights[qc]
                # #     x_qc = cell_triangle.quadrature_points[:, qc]
                # #     v = cell_basis_k.get_phi_vector(x_c, x_qc, h_c)
                # #     fun_grad_proj_val += v @ fun_grad_proj
                # #     grad_check_val += v @ grad_comp @ fun_proj
                # # print("VAL COMP : {} VS {}".format(grad_check_val, fun_grad_proj_val))
                # print(grad_check)
                # print(fun_grad_proj)
                # print("--- FULL GRADIENT")
                # fun_proj = get_element_projection_vector(cell_triangle, faces_segment, fun)
                # # grad_comp = get_full_grad_component_matrix(cell_triangle, faces_segment, dir_x, dir_y)
                # grad_comp = gradop.get_regular_gradient_component_matrix(
                #     field, finite_element, cell_triangle, faces_segment, dir_x, dir_y
                # )
                # fun_grad_proj = get_gradient_projection_vector(cell_triangle, fun_grad_full[choice])
                # grad_check = grad_comp @ fun_proj
                # np.testing.assert_allclose(grad_check, fun_grad_proj, rtol=rtol, atol=atol)
                # # x_c = cell_triangle.shape.centroid
                # # h_c = cell_triangle.shape.diameter
                # # grad_check_val = 0.0
                # # fun_grad_proj_val = 0.0
                # # for qc in range(len(cell_triangle.quadrature_weights)):
                # #     w_qc = cell_triangle.quadrature_weights[qc]
                # #     x_qc = cell_triangle.quadrature_points[:, qc]
                # #     v = cell_basis_k.get_phi_vector(x_c, x_qc, h_c)
                # #     fun_grad_proj_val += v @ fun_grad_proj
                # #     grad_check_val += v @ grad_comp @ fun_proj
                # # print("VAL COMP : {} VS {}".format(grad_check_val, fun_grad_proj_val))
                # print(grad_check)
                # print(fun_grad_proj)

        # # --------------------------------------------------------------------------------------------------------------
        # # DEFINE POLYNOMIAL ORDER AND INTEGRATION ORDER
        # # --------------------------------------------------------------------------------------------------------------
        # face_polynomial_order = 2
        # cell_polynomial_order = face_polynomial_order + 0
        # euclidean_dimension = 2
        # integration_order = 2 * (face_polynomial_order + 1)
        # # integration_order = 2 * (face_polynomial_order+0)
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # DEFINE POLYNOMIAL BASIS
        # # --------------------------------------------------------------------------------------------------------------
        # cell_basis_k = Monomial(face_polynomial_order, euclidean_dimension)
        # cell_basis_l = Monomial(cell_polynomial_order, euclidean_dimension)
        # face_basis_k = Monomial(face_polynomial_order, euclidean_dimension - 1)
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # DEFINE RANDOM POLYNOMIAL COEFFICIENTS
        # # --------------------------------------------------------------------------------------------------------------
        # # range_min = -3.0
        # # range_max = +3.0
        # # coefficients_k = np.array([uniform(range_min, range_max) for _i in range(cell_basis_k.dimension)])
        # # coefficients_l = np.array([uniform(range_min, range_max) for _i in range(cell_basis_l.dimension)])
        # # print("COEFS_K : \n{}".format(coefficients_k))
        # # print("COEFS_L : \n{}".format(coefficients_l))
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # DEFINE MONOMIAL VALUES COMPUTATION
        # # --------------------------------------------------------------------------------------------------------------
        # def test_function(
        #     basis: Monomial, point: ndarray, centroid: ndarray, diameter: float, coefficients: ndarray
        # ) -> float:
        #     value = 0.0
        #     for _i, _exponent in enumerate(basis.exponents):
        #         prod = 1.0
        #         for _x_dir in range(basis.exponents.shape[1]):
        #             prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
        #         prod *= coefficients[_i]
        #         value += prod
        #     return value
        #
        # def test_function_derivative(
        #     basis: Monomial, point: ndarray, centroid: ndarray, diameter: float, direction: int, coefficients: ndarray
        # ) -> float:
        #     value = 0.0
        #     for _i, _exponent in enumerate(basis.exponents):
        #         prod = 1.0
        #         for _x_dir in range(basis.exponents.shape[1]):
        #             if _x_dir == direction:
        #                 prod *= (_exponent[_x_dir] / diameter) * (
        #                     ((point[_x_dir] - centroid[_x_dir]) / diameter) ** (_exponent[_x_dir] - 1)
        #                 )
        #             else:
        #                 prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
        #         prod *= coefficients[_i]
        #         value += prod
        #     return value
        #
        # def get_phi_vector(point: ndarray, centroid: ndarray, diameter: float) -> ndarray:
        #     phi_vector = np.zeros((cell_basis_k.dimension,), dtype=real)
        #     for _i, _exponent in enumerate(cell_basis_k.exponents):
        #         prod = 1.0
        #         for _x_dir in range(cell_basis_k.exponents.shape[1]):
        #             prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
        #         phi_vector[_i] += prod
        #     return phi_vector
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # DEFINE TRIANGLE COORDINATES
        # # --------------------------------------------------------------------------------------------------------------
        # v0 = np.array([1.0, 1.7], dtype=real)
        # v1 = np.array([2.0, 1.6], dtype=real)
        # v2 = np.array([1.9, 3.0], dtype=real)
        # triangle_vertices = np.array([v0, v1, v2], dtype=real).T
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # BUILD CELL
        # # --------------------------------------------------------------------------------------------------------------
        # cell_triangle = Cell(ShapeType.TRIANGLE, triangle_vertices, integration_order)
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # BUILD FACES
        # # --------------------------------------------------------------------------------------------------------------
        # faces_segment = [
        #     Face(ShapeType.SEGMENT, triangle_vertices[:, [0, 1]], integration_order),
        #     Face(ShapeType.SEGMENT, triangle_vertices[:, [1, 2]], integration_order),
        #     Face(ShapeType.SEGMENT, triangle_vertices[:, [2, 0]], integration_order),
        # ]
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # BUILD AND TEST GRADIENT
        # # --------------------------------------------------------------------------------------------------------------
        # def get_sym_grad_component_matrix(cell: Cell, faces: List[Face], _i: int, _j: int):
        #     _d = 2
        #     _dx = 2
        #     _cl = cell_basis_l.dimension
        #     _ck = cell_basis_k.dimension
        #     _fk = face_basis_k.dimension
        #     _nf = len(faces)
        #     _es = _dx * (_cl + _nf * _fk)
        #     _gs = 4
        #     #
        #     x_c = cell.shape.centroid
        #     h_c = cell.shape.diameter
        #     local_grad_matric = np.zeros((_ck, _es), dtype=real)
        #     m_mas = np.zeros((_ck, _ck), dtype=real)
        #     m_adv_j = np.zeros((_ck, _cl), dtype=real)
        #     m_adv_i = np.zeros((_ck, _cl), dtype=real)
        #     for qc in range(len(cell.quadrature_weights)):
        #         x_q_c = cell.quadrature_points[:, qc]
        #         w_q_c = cell.quadrature_weights[qc]
        #         phi_k = cell_basis_k.get_phi_vector(x_q_c, x_c, h_c)
        #         d_phi_l_j = cell_basis_l.get_d_phi_vector(x_q_c, x_c, h_c, _j)
        #         d_phi_l_i = cell_basis_l.get_d_phi_vector(x_q_c, x_c, h_c, _i)
        #         m_adv_j += w_q_c * np.tensordot(phi_k, d_phi_l_j, axes=0)
        #         m_adv_i += w_q_c * np.tensordot(phi_k, d_phi_l_i, axes=0)
        #         m_mas += w_q_c * np.tensordot(phi_k, phi_k, axes=0)
        #     m_mas_inv = np.linalg.inv(m_mas)
        #     _c0 = _i * _cl
        #     _c1 = (_i + 1) * _cl
        #     local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * m_adv_j
        #     _c0 = _j * _cl
        #     _c1 = (_j + 1) * _cl
        #     local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * m_adv_i
        #     for _f, face in enumerate(faces):
        #         h_f = face.shape.diameter
        #         x_f = face.shape.centroid
        #         dist_in_face = (face.mapping_matrix @ (face.shape.centroid - cell.shape.centroid))[-1]
        #         if dist_in_face > 0:
        #             normal_vector_component_j = face.mapping_matrix[-1, _j]
        #             normal_vector_component_i = face.mapping_matrix[-1, _i]
        #         else:
        #             normal_vector_component_j = -face.mapping_matrix[-1, _j]
        #             normal_vector_component_i = -face.mapping_matrix[-1, _i]
        #         m_mas_f = np.zeros((_ck, _cl), dtype=real)
        #         m_hyb_f = np.zeros((_ck, _fk), dtype=real)
        #         for qf in range(len(face.quadrature_weights)):
        #             x_q_f = face.quadrature_points[:, qf]
        #             # x_q_f_prime = face.mapping_matrix @ face.quadrature_points[:, qf]
        #             w_q_f = face.quadrature_weights[qf]
        #             s_f = (face.mapping_matrix @ x_f)[:-1]
        #             s_q_f = (face.mapping_matrix @ x_q_f)[:-1]
        #             phi_k = cell_basis_k.get_phi_vector(x_q_f, x_c, h_c)
        #             phi_l = cell_basis_l.get_phi_vector(x_q_f, x_c, h_c)
        #             # phi_k = cell_basis_k.get_phi_vector(x_q_f_prime, x_c, h_c)
        #             # phi_l = cell_basis_l.get_phi_vector(x_q_f_prime, x_c, h_c)
        #             psi_k = face_basis_k.get_phi_vector(s_q_f, s_f, h_f)
        #             m_mas_f += w_q_f * np.tensordot(phi_k, phi_l, axes=0)
        #             m_hyb_f += w_q_f * np.tensordot(phi_k, psi_k, axes=0)
        #         _c0 = _i * _cl
        #         _c1 = (_i + 1) * _cl
        #         local_grad_matric[:, _c0:_c1] -= (1.0 / 2.0) * m_mas_f * normal_vector_component_j
        #         _c0 = _dx * _cl + _f * _dx * _fk + _i * _fk
        #         _c1 = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
        #         local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * m_hyb_f * normal_vector_component_j
        #         _c0 = _j * _cl
        #         _c1 = (_j + 1) * _cl
        #         local_grad_matric[:, _c0:_c1] -= (1.0 / 2.0) * m_mas_f * normal_vector_component_i
        #         _c0 = _dx * _cl + _f * _dx * _fk + _j * _fk
        #         _c1 = _dx * _cl + _f * _dx * _fk + (_j + 1) * _fk
        #         local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * m_hyb_f * normal_vector_component_i
        #     # local_grad_matric2 = (1.0 / 2.0) * m_mas_inv @ local_grad_matric
        #     invert = True
        #     if invert:
        #         # local_grad_matric2 = m_mas_inv @ ((1.0 / 2.0) * local_grad_matric)
        #         local_grad_matric2 = m_mas_inv @ local_grad_matric
        #     else:
        #         local_grad_matric2 = np.zeros(local_grad_matric.shape, dtype=real)
        #         for col in range(local_grad_matric.shape[1]):
        #             local_grad_matric2[:, col] = (1.0 / 2.0) * np.linalg.solve(m_mas, local_grad_matric[:, col])
        #     print("SYMMETRIC GRADIENT MASS MATRIX CONDITIONING : {}".format(np.linalg.cond(m_mas)))
        #     return local_grad_matric2
        #
        # def get_full_grad_component_matrix(cell: Cell, faces: List[Face], _i: int, _j: int):
        #     _d = 2
        #     _dx = 2
        #     _cl = cell_basis_l.dimension
        #     _ck = cell_basis_k.dimension
        #     _fk = face_basis_k.dimension
        #     _nf = len(faces)
        #     _es = _dx * (_cl + _nf * _fk)
        #     _gs = 4
        #     #
        #     x_c = cell.shape.centroid
        #     h_c = cell.shape.diameter
        #     local_grad_matric = np.zeros((_ck, _es), dtype=real)
        #     m_mas = np.zeros((_ck, _ck), dtype=real)
        #     m_adv_j = np.zeros((_ck, _cl), dtype=real)
        #     for qc in range(len(cell.quadrature_weights)):
        #         x_q_c = cell.quadrature_points[:, qc]
        #         w_q_c = cell.quadrature_weights[qc]
        #         phi_k = cell_basis_k.get_phi_vector(x_q_c, x_c, h_c)
        #         d_phi_l_j = cell_basis_l.get_d_phi_vector(x_q_c, x_c, h_c, _j)
        #         m_adv_j += w_q_c * np.tensordot(phi_k, d_phi_l_j, axes=0)
        #         m_mas += w_q_c * np.tensordot(phi_k, phi_k, axes=0)
        #     m_mas_inv = np.linalg.inv(m_mas)
        #     _c0 = _i * _cl
        #     _c1 = (_i + 1) * _cl
        #     local_grad_matric[:, _c0:_c1] += m_adv_j
        #     for _f, face in enumerate(faces):
        #         h_f = face.shape.diameter
        #         x_f = face.shape.centroid
        #         dist_in_face = (face.mapping_matrix @ (face.shape.centroid - cell.shape.centroid))[-1]
        #         if dist_in_face > 0:
        #             normal_vector_component_j = face.mapping_matrix[-1, _j]
        #         else:
        #             normal_vector_component_j = -face.mapping_matrix[-1, _j]
        #         m_mas_f = np.zeros((_ck, _cl), dtype=real)
        #         m_hyb_f = np.zeros((_ck, _fk), dtype=real)
        #         for qf in range(len(face.quadrature_weights)):
        #             x_q_f = face.quadrature_points[:, qf]
        #             # x_q_f_prime = face.mapping_matrix @ face.quadrature_points[:, qf]
        #             w_q_f = face.quadrature_weights[qf]
        #             s_f = (face.mapping_matrix @ x_f)[:-1]
        #             s_q_f = (face.mapping_matrix @ x_q_f)[:-1]
        #             phi_k = cell_basis_k.get_phi_vector(x_q_f, x_c, h_c)
        #             phi_l = cell_basis_l.get_phi_vector(x_q_f, x_c, h_c)
        #             # phi_k = cell_basis_k.get_phi_vector(x_q_f_prime, x_c, h_c)
        #             # phi_l = cell_basis_l.get_phi_vector(x_q_f_prime, x_c, h_c)
        #             psi_k = face_basis_k.get_phi_vector(s_q_f, s_f, h_f)
        #             m_mas_f += w_q_f * np.tensordot(phi_k, phi_l, axes=0)
        #             m_hyb_f += w_q_f * np.tensordot(phi_k, psi_k, axes=0)
        #         _c0 = _i * _cl
        #         _c1 = (_i + 1) * _cl
        #         local_grad_matric[:, _c0:_c1] -= m_mas_f * normal_vector_component_j
        #         _c0 = _dx * _cl + _f * _dx * _fk + _i * _fk
        #         _c1 = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
        #         local_grad_matric[:, _c0:_c1] += m_hyb_f * normal_vector_component_j
        #     local_grad_matric2 = m_mas_inv @ local_grad_matric
        #     print("FULL GRADIENT MASS MATRIX CONDITIONING : {}".format(np.linalg.cond(m_mas)))
        #     return local_grad_matric2
        #
        # def get_element_projection_vector(cell: Cell, faces: List[Face], function: List[Callable]):
        #     _d = 2
        #     _dx = 2
        #     _cl = cell_basis_l.dimension
        #     _ck = cell_basis_k.dimension
        #     _fk = face_basis_k.dimension
        #     _nf = len(faces)
        #     _es = _dx * (_cl + _nf * _fk)
        #     _gs = 4
        #     #
        #     x_c = cell.shape.centroid
        #     h_c = cell.shape.diameter
        #     matrix = np.zeros((_es, _es), dtype=real)
        #     vector = np.zeros((_es,), dtype=real)
        #     for _dir in range(_dx):
        #         m_mas = np.zeros((_cl, _cl), dtype=real)
        #         vc = np.zeros((_cl,), dtype=real)
        #         for qc in range(len(cell.quadrature_weights)):
        #             x_q_c = cell.quadrature_points[:, qc]
        #             w_q_c = cell.quadrature_weights[qc]
        #             phi_l = cell_basis_l.get_phi_vector(x_q_c, x_c, h_c)
        #             m_mas += w_q_c * np.tensordot(phi_l, phi_l, axes=0)
        #             vc += w_q_c * phi_l * function[_dir](x_q_c)
        #         _i = _cl * _dir
        #         _j = _cl * (_dir + 1)
        #         matrix[_i:_j, _i:_j] += m_mas
        #         vector[_i:_j] += vc
        #     for _f, face in enumerate(faces):
        #         h_f = face.shape.diameter
        #         x_f = face.shape.centroid
        #         for _dir in range(_dx):
        #             m_mas_f = np.zeros((_fk, _fk), dtype=real)
        #             vf = np.zeros((_fk,), dtype=real)
        #             for qf in range(len(face.quadrature_weights)):
        #                 x_q_f = face.quadrature_points[:, qf]
        #                 w_q_f = face.quadrature_weights[qf]
        #                 s_f = (face.mapping_matrix @ x_f)[:-1]
        #                 s_q_f = (face.mapping_matrix @ x_q_f)[:-1]
        #                 psi_k = face_basis_k.get_phi_vector(s_q_f, s_f, h_f)
        #                 m_mas_f += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
        #                 vf += w_q_f * psi_k * function[_dir](x_q_f)
        #             _i = _cl * _dx + _f * _fk * _dx + _dir * _fk
        #             _j = _cl * _dx + _f * _fk * _dx + (_dir + 1) * _fk
        #             matrix[_i:_j, _i:_j] += m_mas_f
        #             vector[_i:_j] += vf
        #     projection_vector = np.linalg.solve(matrix, vector)
        #     return projection_vector
        #
        # def get_gradient_projection_vector(cell: Cell, faces: List[Face], function: Callable):
        #     _d = 2
        #     _dx = 2
        #     _cl = cell_basis_l.dimension
        #     _ck = cell_basis_k.dimension
        #     _fk = face_basis_k.dimension
        #     _nf = len(faces)
        #     _cs = _cl + _nf * _fk
        #     _gs = 4
        #     #
        #     x_c = cell.shape.centroid
        #     h_c = cell.shape.diameter
        #     matrix = np.zeros((_ck, _ck), dtype=real)
        #     vector = np.zeros((_ck,), dtype=real)
        #     for qc in range(len(cell.quadrature_weights)):
        #         x_q_c = cell.quadrature_points[:, qc]
        #         w_q_c = cell.quadrature_weights[qc]
        #         phi_k = cell_basis_k.get_phi_vector(x_q_c, x_c, h_c)
        #         matrix += w_q_c * np.tensordot(phi_k, phi_k, axes=0)
        #         vector += w_q_c * phi_k * function(x_q_c)
        #     projection_vector = np.linalg.solve(matrix, vector)
        #     return projection_vector
        #
        # fun = [lambda x: np.cos(x[0]) * np.sin(x[1]), lambda x: x[0] * x[1]]
        #
        # fun_grad_sym = [
        #     lambda x: -(np.sin(x[0]) * np.sin(x[1])),
        #     lambda x: x[0],
        #     lambda x: (1.0 / 2.0) * (np.cos(x[0]) * np.cos(x[1]) + x[1]),
        #     lambda x: (1.0 / 2.0) * (np.cos(x[0]) * np.cos(x[1]) + x[1]),
        # ]
        #
        # fun_grad_full = [
        #     lambda x: -(np.sin(x[0]) * np.sin(x[1])),
        #     lambda x: x[0],
        #     lambda x: (np.cos(x[0]) * np.cos(x[1])),
        #     lambda x: x[1],
        # ]
        #
        # fun = [lambda x: 2.0 * x[0] + 3.0 * x[1], lambda x: 6.0 * x[0] * x[1]]
        #
        # fun_grad_sym = [
        #     lambda x: 2.0,
        #     lambda x: 6.0 * x[0],
        #     lambda x: (1.0 / 2.0) * (3.0 + 6.0 * x[1]),
        #     lambda x: (1.0 / 2.0) * (3.0 + 6.0 * x[1]),
        # ]
        #
        # fun_grad_full = [lambda x: 2.0, lambda x: 6.0 * x[0], lambda x: 3.0, lambda x: 6.0 * x[1]]
        #
        # correspondance = {0: (0, 0), 1: (1, 1), 2: (0, 1), 3: (1, 0)}
        # choice = 1
        # dir_x = correspondance[choice][0]
        # dir_y = correspondance[choice][1]
        # rtol = 1.0e-3
        # atol = 1.0e-3
        # print("--- SYMMETRIC GRADIENT")
        # fun_proj = get_element_projection_vector(cell_triangle, faces_segment, fun)
        # grad_comp = get_sym_grad_component_matrix(cell_triangle, faces_segment, dir_x, dir_y)
        # fun_grad_proj = get_gradient_projection_vector(cell_triangle, faces_segment, fun_grad_sym[choice])
        # grad_check = grad_comp @ fun_proj
        # np.testing.assert_allclose(grad_check, fun_grad_proj, rtol=rtol, atol=atol)
        # x_c = cell_triangle.shape.centroid
        # h_c = cell_triangle.shape.diameter
        # grad_check_val = 0.0
        # fun_grad_proj_val = 0.0
        # for qc in range(len(cell_triangle.quadrature_weights)):
        #     w_qc = cell_triangle.quadrature_weights[qc]
        #     x_qc = cell_triangle.quadrature_points[:, qc]
        #     v = cell_basis_k.get_phi_vector(x_c, x_qc, h_c)
        #     fun_grad_proj_val += v @ fun_grad_proj
        #     grad_check_val += v @ grad_comp @ fun_proj
        # print("VAL COMP : {} VS {}".format(grad_check_val, fun_grad_proj_val))
        # print(grad_check)
        # print(fun_grad_proj)
        # print("--- FULL GRADIENT")
        # fun_proj = get_element_projection_vector(cell_triangle, faces_segment, fun)
        # grad_comp = get_full_grad_component_matrix(cell_triangle, faces_segment, dir_x, dir_y)
        # fun_grad_proj = get_gradient_projection_vector(cell_triangle, faces_segment, fun_grad_full[choice])
        # grad_check = grad_comp @ fun_proj
        # np.testing.assert_allclose(grad_check, fun_grad_proj, rtol=rtol, atol=atol)
        # x_c = cell_triangle.shape.centroid
        # h_c = cell_triangle.shape.diameter
        # grad_check_val = 0.0
        # fun_grad_proj_val = 0.0
        # for qc in range(len(cell_triangle.quadrature_weights)):
        #     w_qc = cell_triangle.quadrature_weights[qc]
        #     x_qc = cell_triangle.quadrature_points[:, qc]
        #     v = cell_basis_k.get_phi_vector(x_c, x_qc, h_c)
        #     fun_grad_proj_val += v @ fun_grad_proj
        #     grad_check_val += v @ grad_comp @ fun_proj
        # print("VAL COMP : {} VS {}".format(grad_check_val, fun_grad_proj_val))
        # print(grad_check)
        # print(fun_grad_proj)
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # BUILD AND TEST STABILIZATION
        # # --------------------------------------------------------------------------------------------------------------
        #
        # def get_stabilization_operator(cell: Cell, faces: List[Face]):
        #     _dx = 2
        #     _cl = cell_basis_l.dimension
        #     _fk = face_basis_k.dimension
        #     _nf = len(faces)
        #     _es = _dx * (_cl + _nf * _fk)
        #     stabilization_operator = np.zeros((_es, _es), dtype=real)
        #     stabilization_operator2 = np.zeros((_fk, _es), dtype=real)
        #     stabilization_operator_0 = np.zeros((_es, _es), dtype=real)
        #     x_c = cell.shape.centroid
        #     h_c = cell.shape.diameter
        #     for _f, face in enumerate(faces):
        #         h_f = face.shape.diameter
        #         x_f = face.shape.centroid
        #         stabilization_op = np.zeros((_fk, _es), dtype=real)
        #         stabilization_op_0 = np.zeros((_dx * _fk, _es), dtype=real)
        #         m_mas_f = np.zeros((_fk, _fk), dtype=real)
        #         m_mas_f_0 = np.zeros((_dx * _fk, _dx * _fk), dtype=real)
        #         m_hyb_f = np.zeros((_fk, _cl), dtype=real)
        #         for qf in range(len(face.quadrature_weights)):
        #             x_q_f = face.quadrature_points[:, qf]
        #             w_q_f = face.quadrature_weights[qf]
        #             s_f = (face.mapping_matrix @ x_f)[:-1]
        #             s_q_f = (face.mapping_matrix @ x_q_f)[:-1]
        #             x_q_f_p = face.mapping_matrix @ x_q_f
        #             x_c_p = face.mapping_matrix @ x_c
        #             phi_l = cell_basis_l.get_phi_vector(x_q_f, x_c, h_c)
        #             # phi_l = cell_basis_l.get_phi_vector(x_q_f_p, x_c_p, h_c)
        #             psi_k = face_basis_k.get_phi_vector(s_q_f, s_f, h_f)
        #             m_hyb_f += w_q_f * np.tensordot(psi_k, phi_l, axes=0)
        #             m_mas_f += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
        #             for _x_dir in range(_dx):
        #                 _a0 = _x_dir * _fk
        #                 _a1 = (_x_dir + 1) * _fk
        #                 m_mas_f_0[_a0:_a1, _a0:_a1] += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
        #         m_mas_f_inv = np.linalg.inv(m_mas_f)
        #         proj_mat = m_mas_f_inv @ m_hyb_f
        #         for _x_dir in range(_dx):
        #             _i = _x_dir * _cl
        #             _j = (_x_dir + 1) * _cl
        #             stabilization_op[:, _i:_j] -= proj_mat
        #             _a0 = _x_dir * _fk
        #             _a1 = (_x_dir + 1) * _fk
        #             stabilization_op_0[_a0:_a1, _i:_j] -= proj_mat
        #             _i = _dx * _cl + _f * _dx * _fk + _x_dir * _fk
        #             _j = _dx * _cl + _f * _dx * _fk + (_x_dir + 1) * _fk
        #             m = np.eye(_fk, dtype=real)
        #             stabilization_op[:, _i:_j] += m
        #             _a0 = _x_dir * _fk
        #             _a1 = (_x_dir + 1) * _fk
        #             stabilization_op_0[_a0:_a1, _i:_j] += m
        #         stabilization_operator += (1.0 / h_f) * stabilization_op.T @ m_mas_f @ stabilization_op
        #         stabilization_operator_0 += (1.0 / h_f) * stabilization_op_0.T @ m_mas_f_0 @ stabilization_op_0
        #         stabilization_operator2 += (1.0 / h_f) * stabilization_op
        #     return stabilization_operator, stabilization_operator_0, stabilization_operator2
        #
        # def get_stabilization_operator_component(cell: Cell, faces: List[Face], _i: int):
        #     _dx = 2
        #     _cl = cell_basis_l.dimension
        #     _fk = face_basis_k.dimension
        #     _nf = len(faces)
        #     _es = _dx * (_cl + _nf * _fk)
        #     x_c = cell.shape.centroid
        #     h_c = cell.shape.diameter
        #     stabilization_operator = np.zeros((_es, _es), dtype=real)
        #     stabilization_op = np.zeros((_fk, _es), dtype=real)
        #     for _f, face in enumerate(faces):
        #         h_f = face.shape.diameter
        #         x_f = face.shape.centroid
        #         m_mas_f = np.zeros((_fk, _fk), dtype=real)
        #         m_hyb_f = np.zeros((_fk, _cl), dtype=real)
        #         for qf in range(len(face.quadrature_weights)):
        #             x_q_f = face.quadrature_points[:, qf]
        #             w_q_f = face.quadrature_weights[qf]
        #             s_f = (face.mapping_matrix @ x_f)[:-1]
        #             s_q_f = (face.mapping_matrix @ x_q_f)[:-1]
        #             x_q_f_p = face.mapping_matrix @ x_q_f
        #             x_c_p = face.mapping_matrix @ x_c
        #             phi_l = cell_basis_l.get_phi_vector(x_q_f, x_c, h_c)
        #             # phi_l = cell_basis_l.get_phi_vector(x_q_f_p, x_c_p, h_c)
        #             psi_k = face_basis_k.get_phi_vector(s_q_f, s_f, h_f)
        #             m_hyb_f += w_q_f * np.tensordot(psi_k, phi_l, axes=0)
        #             m_mas_f += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
        #         m_mas_f_inv = np.linalg.inv(m_mas_f)
        #         proj_mat = m_mas_f_inv @ m_hyb_f
        #         m = np.eye(_fk, dtype=real)
        #         _ci = _i * _cl
        #         _cj = (_i + 1) * _cl
        #         stabilization_op[:, _ci:_cj] -= proj_mat
        #         _ci = _dx * _cl + _f * _dx * _fk + _i * _fk
        #         _cj = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
        #         stabilization_op[:, _ci:_cj] += m
        #         stabilization_operator += (1.0 / h_f) * stabilization_op.T @ m_mas_f @ stabilization_op
        #     return stabilization_operator, stabilization_op
        #
        # fun_proj = get_element_projection_vector(cell_triangle, faces_segment, fun)
        # # stab_matrix, stab_matrix_0, stab_matrix2 = get_stabilization_operator(cell_triangle, faces_segment)
        # stabilization_operator, stabilization_op = get_stabilization_operator_component(cell_triangle, faces_segment, 1)
        # print("FUN PROJ")
        # print(fun_proj)
        # print("STABILIZATION")
        # stab_vector = stabilization_op @ fun_proj
        # print(stab_vector)
        # print("FUN PROJ")
        # print(fun_proj)
        # print("STABILIZATION --- OLD")
        # stab_vector = stab_matrix @ fun_proj
        # stab_val = fun_proj @ stab_matrix @ fun_proj
        # print(stab_vector)
        # print(stab_val)
        # print("STABILIZATION --- NEW")
        # stab_vector = stab_matrix_0 @ fun_proj
        # stab_val = fun_proj @ stab_matrix_0 @ fun_proj
        # print(stab_vector)
        # print(stab_val)
        return
