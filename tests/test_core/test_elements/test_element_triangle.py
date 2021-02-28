from unittest import TestCase

from parameterized import parameterized
from scipy import integrate
import quadpy
from random import randrange
from random import uniform

from h2o.h2o import *
from h2o.geometry.shapes.segment import *
from h2o.geometry.shape import Shape
from h2o.fem.element.face import Face
from h2o.fem.element.cell import Cell
from h2o.fem.basis.basis import Basis
from h2o.fem.basis.bases.monomial import Monomial

import matplotlib.pyplot as plt

np.set_printoptions(precision=16)


class TestElementTriangle(TestCase):
    def test_face_segment(self):

        # --------------------------------------------------------------------------------------------------------------
        # DEFINE POLYNOMIAL ORDER AND INTEGRATION ORDER
        # --------------------------------------------------------------------------------------------------------------
        polynomial_order = 2
        euclidean_dimension = 1
        integration_order = 2 * (polynomial_order + 1)

        # --------------------------------------------------------------------------------------------------------------
        # DEFINE POLYNOMIAL BASIS
        # --------------------------------------------------------------------------------------------------------------
        face_basis_k = Monomial(polynomial_order, euclidean_dimension)

        # --------------------------------------------------------------------------------------------------------------
        # DEFINE RANDOM POLYNOMIAL COEFFICIENTS
        # --------------------------------------------------------------------------------------------------------------
        range_min = -3.0
        range_max = +3.0
        coefficients = np.array([uniform(range_min, range_max) for _i in range(face_basis_k.dimension)])
        print("COEFS : \n{}".format(coefficients))

        # --------------------------------------------------------------------------------------------------------------
        # DEFINE MONOMIAL VALUES COMPUTATION
        # --------------------------------------------------------------------------------------------------------------
        def test_function(
            basis: Monomial, point: ndarray, centroid: ndarray, diameter: float, coefficients: ndarray
        ) -> float:
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
            basis: Monomial, point: ndarray, centroid: ndarray, diameter: float, direction: int, coefficients: ndarray
        ) -> float:
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

        def get_phi_vector(point: ndarray, centroid: ndarray, diameter: float) -> ndarray:
            phi_vector = np.zeros((face_basis_k.dimension,), dtype=real)
            for _i, _exponent in enumerate(face_basis_k.exponents):
                prod = 1.0
                for _x_dir in range(face_basis_k.exponents.shape[1]):
                    prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
                phi_vector[_i] += prod
            return phi_vector

        # --------------------------------------------------------------------------------------------------------------
        # DEFINE SEGMENT COORDINATES
        # --------------------------------------------------------------------------------------------------------------
        v_0 = np.array([1.0, 1.0], dtype=real)
        v_1 = np.array([2.0, 5.0], dtype=real)
        segment_vertices = np.array([v_0, v_1], dtype=real).T

        # --------------------------------------------------------------------------------------------------------------
        # BUILD FACE
        # --------------------------------------------------------------------------------------------------------------
        face_segment = Face(ShapeType.SEGMENT, segment_vertices, integration_order)
        x_f = face_segment.shape.centroid
        h_f = face_segment.shape.diameter
        # --- PROJECT ON HYPERPLANE
        s_f = (face_segment.mapping_matrix @ face_segment.shape.centroid)[:-1]
        s_0 = (face_segment.mapping_matrix @ v_0)[:-1]
        s_1 = (face_segment.mapping_matrix @ v_1)[:-1]

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
        for _qf in range(len(face_segment.quadrature_weights)):
            x_qf = face_segment.quadrature_points[:, _qf]
            # --- PLOT QUADRATURE POINT
            plt.scatter(x_qf[0], x_qf[1], c="g")
            # --- PLOT PROJECTED QUADRATURE POINT
            s_qf = (face_segment.mapping_matrix @ x_qf)[:-1]
            plt.scatter(s_qf, 0.0, c="grey")
        # --- PRINT QUADRATURE POINTS AND WEIGHTS
        for _qf in range(len(face_segment.quadrature_weights)):
            x_qf = face_segment.quadrature_points[:, _qf]
            w_qf = face_segment.quadrature_weights[_qf]
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
        # CHECK INTEGRATION IN CELL
        # --------------------------------------------------------------------------------------------------------------
        scheme = quadpy.c1.gauss_legendre(2 * integration_order)
        for _i in range(euclidean_dimension):
            for _j in range(euclidean_dimension):
                mass_mat = np.zeros((face_basis_k.dimension, face_basis_k.dimension), dtype=real)
                stif_mat = np.zeros((face_basis_k.dimension, face_basis_k.dimension), dtype=real)
                advc_mat = np.zeros((face_basis_k.dimension, face_basis_k.dimension), dtype=real)
                for _qf in range(len(face_segment.quadrature_weights)):
                    _x_qf = face_segment.quadrature_points[:, _qf]
                    _s_qf = (face_segment.mapping_matrix @ _x_qf)[:-1]
                    _w_qf = face_segment.quadrature_weights[_qf]
                    phi_0 = face_basis_k.get_phi_vector(_s_qf, s_f, h_f)
                    d_phi_0_i = face_basis_k.get_d_phi_vector(_s_qf, s_f, h_f, _i)
                    d_phi_0_j = face_basis_k.get_d_phi_vector(_s_qf, s_f, h_f, _j)
                    mass_mat += _w_qf * np.tensordot(phi_0, phi_0, axes=0)
                    stif_mat += _w_qf * np.tensordot(d_phi_0_i, d_phi_0_j, axes=0)
                    advc_mat += _w_qf * np.tensordot(phi_0, d_phi_0_i, axes=0)
                mass_integral = coefficients @ mass_mat @ coefficients
                stif_integral = coefficients @ stif_mat @ coefficients
                advc_integral = coefficients @ advc_mat @ coefficients
                f_mass_check = lambda x: test_function(
                    face_basis_k, np.array([x]), s_f, h_f, coefficients
                ) * test_function(face_basis_k, np.array([x]), s_f, h_f, coefficients)
                f_stif_check = lambda x: test_function_derivative(
                    face_basis_k, np.array([x]), s_f, h_f, _i, coefficients
                ) * test_function_derivative(face_basis_k, np.array([x]), s_f, h_f, _j, coefficients)
                f_advc_check = lambda x: test_function(
                    face_basis_k, np.array([x]), s_f, h_f, coefficients
                ) * test_function_derivative(face_basis_k, np.array([x]), s_f, h_f, _j, coefficients)
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
        # print(mass_mat)

        # # --------------------------------------------------------------------------------------------------------------
        # # COMPUTE PHI VECTOR INTEGRAL
        # # --------------------------------------------------------------------------------------------------------------
        # # --- COMPUTE INTEGRAL
        # computed_val = 0.0
        # for _qf in range(len(face_segment.quadrature_weights)):
        #     x_qf = face_segment.quadrature_points[:, _qf]
        #     w_qf = face_segment.quadrature_weights[_qf]
        #     s_qf = (face_segment.mapping_matrix @ x_qf)[:-1]
        #     # computed_val += w_qf * get_phi_vector(s_qf, s_f, h_f) @ coefficients
        #     computed_val += w_qf * face_basis_k.get_phi_vector(s_qf, s_f, h_f) @ coefficients
        # # --- GET QUADPY INTEGRAL EVALUATION
        # scheme = quadpy.c1.gauss_legendre(polynomial_order)
        # p_0 = s_0[0]
        # p_1 = s_1[0]
        # val = scheme.integrate(lambda x: test_function(np.array([x]), s_f, h_f, coefficients), [p_0, p_1])
        # print("num_res : {}".format(computed_val))
        # print("val_res : {}".format(val))
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # COMPUTE MASS MATRIX INTEGRAL
        # # --------------------------------------------------------------------------------------------------------------
        # # --- COMPUTE INTEGRAL
        # computed_val = 0.0
        # for _qf in range(len(face_segment.quadrature_weights)):
        #     x_qf = face_segment.quadrature_points[:, _qf]
        #     w_qf = face_segment.quadrature_weights[_qf]
        #     s_qf = (face_segment.mapping_matrix @ x_qf)[:-1]
        #     phi0 = face_basis_k.get_phi_vector(s_qf, s_f, h_f)
        #     phi1 = face_basis_k.get_phi_vector(s_qf, s_f, h_f)
        #     mat = np.tensordot(phi0, phi1, axes=0)
        #     computed_val += w_qf * coefficients @ mat @ coefficients
        # # --- GET QUADPY INTEGRAL EVALUATION
        # scheme = quadpy.c1.gauss_legendre(integration_order)
        # p_0 = s_0[0]
        # p_1 = s_1[0]
        # val = scheme.integrate(lambda x: test_function(np.array([x]), s_f, h_f, coefficients) ** 2, [p_0, p_1])
        # print("num_res : {}".format(computed_val))
        # print("val_res : {}".format(val))

    def test_cell_triangle(self):

        # --------------------------------------------------------------------------------------------------------------
        # DEFINE POLYNOMIAL ORDER AND INTEGRATION ORDER
        # --------------------------------------------------------------------------------------------------------------
        face_polynomial_order = 2
        cell_polynomial_order = face_polynomial_order + 1
        euclidean_dimension = 2
        integration_order = 2 * (face_polynomial_order + 1)

        # --------------------------------------------------------------------------------------------------------------
        # DEFINE POLYNOMIAL BASIS
        # --------------------------------------------------------------------------------------------------------------
        cell_basis_k = Monomial(face_polynomial_order, euclidean_dimension)
        cell_basis_l = Monomial(cell_polynomial_order, euclidean_dimension)

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
            basis: Monomial, point: ndarray, centroid: ndarray, diameter: float, coefficients: ndarray
        ) -> float:
            value = 0.0
            for _i, _exponent in enumerate(basis.exponents):
                prod = 1.0
                for _x_dir in range(basis.exponents.shape[1]):
                    prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
                prod *= coefficients[_i]
                value += prod
            return value

        def test_function_derivative(
            basis: Monomial, point: ndarray, centroid: ndarray, diameter: float, direction: int, coefficients: ndarray
        ) -> float:
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

        def get_phi_vector(point: ndarray, centroid: ndarray, diameter: float) -> ndarray:
            phi_vector = np.zeros((cell_basis_k.dimension,), dtype=real)
            for _i, _exponent in enumerate(cell_basis_k.exponents):
                prod = 1.0
                for _x_dir in range(cell_basis_k.exponents.shape[1]):
                    prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
                phi_vector[_i] += prod
            return phi_vector

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
        cell_triangle = Cell(ShapeType.TRIANGLE, triangle_vertices, integration_order)
        x_c = cell_triangle.shape.centroid
        h_c = cell_triangle.shape.diameter

        # --------------------------------------------------------------------------------------------------------------
        # PLOT CELL
        # --------------------------------------------------------------------------------------------------------------
        # --- PLOT VERTICES AND CENTROID
        plt.scatter(v0[0], v0[1], c="b")
        plt.scatter(v1[0], v1[1], c="b")
        plt.scatter(v2[0], v2[1], c="b")
        plt.scatter(x_c[0], x_c[1], c="b")
        # --- PLOT QUADRATURE POINTS
        for _qc in range(len(cell_triangle.quadrature_weights)):
            _x_qc = cell_triangle.quadrature_points[:, _qc]
            plt.scatter(_x_qc[0], _x_qc[1], c="g")
        # --- PRINT QUADRATURE POINTS AND WEIGHTS
        for _qc in range(len(cell_triangle.quadrature_weights)):
            _x_qc = cell_triangle.quadrature_points[:, _qc]
            _w_qc = cell_triangle.quadrature_weights[_qc]
            print("QUAD_POINT : {} | QUAD_WEIGHT : {}".format(_x_qc, _w_qc))
        # --- SET PLOT OPTIONS
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid()
        plt.show()

        # --------------------------------------------------------------------------------------------------------------
        # CHECK INTEGRATION IN CELL
        # --------------------------------------------------------------------------------------------------------------
        bases = [cell_basis_k, cell_basis_l]
        orders = [face_polynomial_order, cell_polynomial_order]
        coefs = [coefficients_k, coefficients_l]
        scheme = quadpy.t2.get_good_scheme(2 * integration_order)
        for basis_0, order_0, coef_0 in zip(bases, orders, coefs):
            for basis_1, order_1, coef_1 in zip(bases, orders, coefs):
                for _i in range(euclidean_dimension):
                    for _j in range(euclidean_dimension):
                        mass_mat = np.zeros((basis_0.dimension, basis_1.dimension), dtype=real)
                        stif_mat = np.zeros((basis_0.dimension, basis_1.dimension), dtype=real)
                        advc_mat = np.zeros((basis_0.dimension, basis_1.dimension), dtype=real)
                        for _qc in range(len(cell_triangle.quadrature_weights)):
                            _x_qc = cell_triangle.quadrature_points[:, _qc]
                            _w_qc = cell_triangle.quadrature_weights[_qc]
                            phi_0 = basis_0.get_phi_vector(_x_qc, x_c, h_c)
                            phi_1 = basis_1.get_phi_vector(_x_qc, x_c, h_c)
                            d_phi_0_i = basis_0.get_d_phi_vector(_x_qc, x_c, h_c, _i)
                            d_phi_1_j = basis_1.get_d_phi_vector(_x_qc, x_c, h_c, _j)
                            mass_mat += _w_qc * np.tensordot(phi_0, phi_1, axes=0)
                            stif_mat += _w_qc * np.tensordot(d_phi_0_i, d_phi_1_j, axes=0)
                            advc_mat += _w_qc * np.tensordot(phi_0, d_phi_1_j, axes=0)
                        mass_integral = coef_0 @ mass_mat @ coef_1
                        stif_integral = coef_0 @ stif_mat @ coef_1
                        advc_integral = coef_0 @ advc_mat @ coef_1
                        f_mass_check = lambda x: test_function(basis_0, x, x_c, h_c, coef_0) * test_function(
                            basis_1, x, x_c, h_c, coef_1
                        )
                        f_stif_check = lambda x: test_function_derivative(
                            basis_0, x, x_c, h_c, _i, coef_0
                        ) * test_function_derivative(basis_1, x, x_c, h_c, _j, coef_1)
                        f_advc_check = lambda x: test_function(basis_0, x, x_c, h_c, coef_0) * test_function_derivative(
                            basis_1, x, x_c, h_c, _j, coef_1
                        )
                        mass_integral_check = scheme.integrate(f_mass_check, triangle_vertices.T)
                        stif_integral_check = scheme.integrate(f_stif_check, triangle_vertices.T)
                        advc_integral_check = scheme.integrate(f_advc_check, triangle_vertices.T)
                        rtol = 1.0e-15
                        atol = 1.0e-15
                        np.testing.assert_allclose(mass_integral_check, mass_integral, rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stif_integral_check, stif_integral, rtol=rtol, atol=atol)
                        np.testing.assert_allclose(advc_integral_check, advc_integral, rtol=rtol, atol=atol)

        # # --- GETTING FACE ORDER POLYNOMIAL UNKNOWNS IN CELL
        # for key_k, val_k in pols.items():
        #     order_k = key_k
        #     cell_basis_k = val_k[0]
        #     coefficients_k = val_k[1]
        #     label_k = val_k[2]
        #     # --- GETTING CELL ORDER POLYNOMIAL UNKNOWNS IN CELL
        #     for key_l, val_l in pols.items():
        #         order_l = key_l
        #         cell_basis_l = val_l[0]
        #         coefficients_l = val_l[1]
        #         label_l = val_l[2]
        # --- GETTING DERIVATIVE DIRECTIONS IN CELL
        # for _i in range(euclidean_dimension):
        #     for _j in range(euclidean_dimension):
        #     _i = 0
        #     _j = 1
        #     # --- INITIALIZING NUMERICAL INTEGRALS
        #     phi_k_check = 0.0
        #     phi_l_check = 0.0
        #     # ---
        #     d_phi_k_i_check = 0.0
        #     d_phi_l_i_check = 0.0
        #     d_phi_k_j_check = 0.0
        #     d_phi_l_j_check = 0.0
        #     # ---
        #     mass_kk_check = 0.0
        #     mass_kl_check = 0.0
        #     mass_lk_check = 0.0
        #     mass_ll_check = 0.0
        #     # ---
        #     stif_kk_ii_check = 0.0
        #     stif_kl_ii_check = 0.0
        #     stif_lk_ii_check = 0.0
        #     stif_ll_ii_check = 0.0
        #     # ---
        #     stif_kk_ij_check = 0.0
        #     stif_kl_ij_check = 0.0
        #     stif_lk_ij_check = 0.0
        #     stif_ll_ij_check = 0.0
        #     # ---
        #     stif_kk_ji_check = 0.0
        #     stif_kl_ji_check = 0.0
        #     stif_lk_ji_check = 0.0
        #     stif_ll_ji_check = 0.0
        #     # ---
        #     stif_kk_jj_check = 0.0
        #     stif_kl_jj_check = 0.0
        #     stif_lk_jj_check = 0.0
        #     stif_ll_jj_check = 0.0
        #     # ---
        #     advc_kk_i_check = 0.0
        #     advc_kl_i_check = 0.0
        #     advc_lk_i_check = 0.0
        #     advc_ll_i_check = 0.0
        #     # ---
        #     advc_kk_j_check = 0.0
        #     advc_kl_j_check = 0.0
        #     advc_lk_j_check = 0.0
        #     advc_ll_j_check = 0.0
        #     # --- DEFINING ANALYTICAL FUNCTIONS TO PASS TO QUADPY
        #     f_phi_k_check = lambda x: test_function(cell_basis_k, x, x_c, h_c, coefficients_k)
        #     f_phi_l_check = lambda x: test_function(cell_basis_l, x, x_c, h_c, coefficients_l)
        #     # ---
        #     f_d_phi_k_i_check = lambda x: test_function_derivative(cell_basis_k, x, x_c, h_c, _i, coefficients_k)
        #     f_d_phi_l_i_check = lambda x: test_function_derivative(cell_basis_l, x, x_c, h_c, _i, coefficients_l)
        #     f_d_phi_k_j_check = lambda x: test_function_derivative(cell_basis_k, x, x_c, h_c, _j, coefficients_k)
        #     f_d_phi_l_j_check = lambda x: test_function_derivative(cell_basis_l, x, x_c, h_c, _j, coefficients_l)
        #     # ---
        #     f_mass_kk_check = lambda x: test_function(cell_basis_k, x, x_c, h_c, coefficients_k) * test_function(
        #         cell_basis_k, x, x_c, h_c, coefficients_k
        #     )
        #     f_mass_kl_check = lambda x: test_function(cell_basis_k, x, x_c, h_c, coefficients_k) * test_function(
        #         cell_basis_l, x, x_c, h_c, coefficients_l
        #     )
        #     f_mass_lk_check = lambda x: test_function(cell_basis_l, x, x_c, h_c, coefficients_l) * test_function(
        #         cell_basis_k, x, x_c, h_c, coefficients_k
        #     )
        #     f_mass_ll_check = lambda x: test_function(cell_basis_l, x, x_c, h_c, coefficients_l) * test_function(
        #         cell_basis_l, x, x_c, h_c, coefficients_l
        #     )
        #     # ---
        #     f_stif_kk_ii_check = lambda x: test_function_derivative(
        #         cell_basis_k, x, x_c, h_c, _i, coefficients_k
        #     ) * test_function_derivative(cell_basis_k, x, x_c, h_c, _i, coefficients_k)
        #     f_stif_kl_ii_check = lambda x: test_function_derivative(
        #         cell_basis_k, x, x_c, h_c, _i, coefficients_k
        #     ) * test_function_derivative(cell_basis_l, x, x_c, h_c, _i, coefficients_l)
        #     f_stif_lk_ii_check = lambda x: test_function_derivative(
        #         cell_basis_l, x, x_c, h_c, _i, coefficients_l
        #     ) * test_function_derivative(cell_basis_k, x, x_c, h_c, _i, coefficients_k)
        #     f_stif_ll_ii_check = lambda x: test_function_derivative(
        #         cell_basis_l, x, x_c, h_c, _i, coefficients_l
        #     ) * test_function_derivative(cell_basis_l, x, x_c, h_c, _i, coefficients_l)
        #     # ---
        #     f_stif_kk_ij_check = lambda x: test_function_derivative(
        #         cell_basis_k, x, x_c, h_c, _i, coefficients_k
        #     ) * test_function_derivative(cell_basis_k, x, x_c, h_c, _j, coefficients_k)
        #     f_stif_kl_ij_check = lambda x: test_function_derivative(
        #         cell_basis_k, x, x_c, h_c, _i, coefficients_k
        #     ) * test_function_derivative(cell_basis_l, x, x_c, h_c, _j, coefficients_l)
        #     f_stif_lk_ij_check = lambda x: test_function_derivative(
        #         cell_basis_l, x, x_c, h_c, _i, coefficients_l
        #     ) * test_function_derivative(cell_basis_k, x, x_c, h_c, _j, coefficients_k)
        #     f_stif_ll_ij_check = lambda x: test_function_derivative(
        #         cell_basis_l, x, x_c, h_c, _i, coefficients_l
        #     ) * test_function_derivative(cell_basis_l, x, x_c, h_c, _j, coefficients_l)
        #     # ---
        #     f_stif_kk_ji_check = lambda x: test_function_derivative(
        #         cell_basis_k, x, x_c, h_c, _j, coefficients_k
        #     ) * test_function_derivative(cell_basis_k, x, x_c, h_c, _i, coefficients_k)
        #     f_stif_kl_ji_check = lambda x: test_function_derivative(
        #         cell_basis_k, x, x_c, h_c, _j, coefficients_k
        #     ) * test_function_derivative(cell_basis_l, x, x_c, h_c, _i, coefficients_l)
        #     f_stif_lk_ji_check = lambda x: test_function_derivative(
        #         cell_basis_l, x, x_c, h_c, _j, coefficients_l
        #     ) * test_function_derivative(cell_basis_k, x, x_c, h_c, _i, coefficients_k)
        #     f_stif_ll_ji_check = lambda x: test_function_derivative(
        #         cell_basis_l, x, x_c, h_c, _j, coefficients_l
        #     ) * test_function_derivative(cell_basis_l, x, x_c, h_c, _i, coefficients_l)
        #     # ---
        #     f_stif_kk_jj_check = lambda x: test_function_derivative(
        #         cell_basis_k, x, x_c, h_c, _j, coefficients_k
        #     ) * test_function_derivative(cell_basis_k, x, x_c, h_c, _j, coefficients_k)
        #     f_stif_kl_jj_check = lambda x: test_function_derivative(
        #         cell_basis_k, x, x_c, h_c, _j, coefficients_k
        #     ) * test_function_derivative(cell_basis_l, x, x_c, h_c, _j, coefficients_l)
        #     f_stif_lk_jj_check = lambda x: test_function_derivative(
        #         cell_basis_l, x, x_c, h_c, _j, coefficients_l
        #     ) * test_function_derivative(cell_basis_k, x, x_c, h_c, _j, coefficients_k)
        #     f_stif_ll_jj_check = lambda x: test_function_derivative(
        #         cell_basis_l, x, x_c, h_c, _j, coefficients_l
        #     ) * test_function_derivative(cell_basis_l, x, x_c, h_c, _j, coefficients_l)
        #     # ---
        #     f_advc_kk_i_check = lambda x: test_function(
        #         cell_basis_k, x, x_c, h_c, coefficients_k
        #     ) * test_function_derivative(cell_basis_k, x, x_c, h_c, _i, coefficients_k)
        #     f_advc_kl_i_check = lambda x: test_function(
        #         cell_basis_k, x, x_c, h_c, coefficients_k
        #     ) * test_function_derivative(cell_basis_l, x, x_c, h_c, _i, coefficients_l)
        #     f_advc_lk_i_check = lambda x: test_function(
        #         cell_basis_l, x, x_c, h_c, coefficients_l
        #     ) * test_function_derivative(cell_basis_k, x, x_c, h_c, _i, coefficients_k)
        #     f_advc_ll_i_check = lambda x: test_function(
        #         cell_basis_l, x, x_c, h_c, coefficients_l
        #     ) * test_function_derivative(cell_basis_l, x, x_c, h_c, _i, coefficients_l)
        #     # ---
        #     f_advc_kk_j_check = lambda x: test_function(
        #         cell_basis_k, x, x_c, h_c, coefficients_k
        #     ) * test_function_derivative(cell_basis_k, x, x_c, h_c, _j, coefficients_k)
        #     f_advc_kl_j_check = lambda x: test_function(
        #         cell_basis_k, x, x_c, h_c, coefficients_k
        #     ) * test_function_derivative(cell_basis_l, x, x_c, h_c, _j, coefficients_l)
        #     f_advc_lk_j_check = lambda x: test_function(
        #         cell_basis_l, x, x_c, h_c, coefficients_l
        #     ) * test_function_derivative(cell_basis_k, x, x_c, h_c, _j, coefficients_k)
        #     f_advc_ll_j_check = lambda x: test_function(
        #         cell_basis_l, x, x_c, h_c, coefficients_l
        #     ) * test_function_derivative(cell_basis_l, x, x_c, h_c, _j, coefficients_l)
        #     # --- INITIALIZING QUADPY RESULTS
        #     quadpy_phi_k_check = scheme.integrate(f_phi_k_check, triangle_vertices.T)
        #     quadpy_phi_l_check = scheme.integrate(f_phi_l_check, triangle_vertices.T)
        #     # ---
        #     quadpy_d_phi_k_i_check = scheme.integrate(f_d_phi_k_i_check, triangle_vertices.T)
        #     quadpy_d_phi_l_i_check = scheme.integrate(f_d_phi_l_i_check, triangle_vertices.T)
        #     quadpy_d_phi_k_j_check = scheme.integrate(f_d_phi_k_j_check, triangle_vertices.T)
        #     quadpy_d_phi_l_j_check = scheme.integrate(f_d_phi_l_j_check, triangle_vertices.T)
        #     # ---
        #     quadpy_mass_kk_check = scheme.integrate(f_mass_kk_check, triangle_vertices.T)
        #     quadpy_mass_kl_check = scheme.integrate(f_mass_kl_check, triangle_vertices.T)
        #     quadpy_mass_lk_check = scheme.integrate(f_mass_lk_check, triangle_vertices.T)
        #     quadpy_mass_ll_check = scheme.integrate(f_mass_ll_check, triangle_vertices.T)
        #     # ---
        #     quadpy_stif_kk_ii_check = scheme.integrate(f_stif_kk_ii_check, triangle_vertices.T)
        #     quadpy_stif_kl_ii_check = scheme.integrate(f_stif_kl_ii_check, triangle_vertices.T)
        #     quadpy_stif_lk_ii_check = scheme.integrate(f_stif_lk_ii_check, triangle_vertices.T)
        #     quadpy_stif_ll_ii_check = scheme.integrate(f_stif_ll_ii_check, triangle_vertices.T)
        #     # ---
        #     quadpy_stif_kk_ij_check = scheme.integrate(f_stif_kk_ij_check, triangle_vertices.T)
        #     quadpy_stif_kl_ij_check = scheme.integrate(f_stif_kl_ij_check, triangle_vertices.T)
        #     quadpy_stif_lk_ij_check = scheme.integrate(f_stif_lk_ij_check, triangle_vertices.T)
        #     quadpy_stif_ll_ij_check = scheme.integrate(f_stif_ll_ij_check, triangle_vertices.T)
        #     # ---
        #     quadpy_stif_kk_ji_check = scheme.integrate(f_stif_kk_ji_check, triangle_vertices.T)
        #     quadpy_stif_kl_ji_check = scheme.integrate(f_stif_kl_ji_check, triangle_vertices.T)
        #     quadpy_stif_lk_ji_check = scheme.integrate(f_stif_lk_ji_check, triangle_vertices.T)
        #     quadpy_stif_ll_ji_check = scheme.integrate(f_stif_ll_ji_check, triangle_vertices.T)
        #     # ---
        #     quadpy_stif_kk_jj_check = scheme.integrate(f_stif_kk_jj_check, triangle_vertices.T)
        #     quadpy_stif_kl_jj_check = scheme.integrate(f_stif_kl_jj_check, triangle_vertices.T)
        #     quadpy_stif_lk_jj_check = scheme.integrate(f_stif_lk_jj_check, triangle_vertices.T)
        #     quadpy_stif_ll_jj_check = scheme.integrate(f_stif_ll_jj_check, triangle_vertices.T)
        #     # ---
        #     quadpy_advc_kk_i_check = scheme.integrate(f_advc_kk_i_check, triangle_vertices.T)
        #     quadpy_advc_kl_i_check = scheme.integrate(f_advc_kl_i_check, triangle_vertices.T)
        #     quadpy_advc_lk_i_check = scheme.integrate(f_advc_lk_i_check, triangle_vertices.T)
        #     quadpy_advc_ll_i_check = scheme.integrate(f_advc_ll_i_check, triangle_vertices.T)
        #     # ---
        #     quadpy_advc_kk_j_check = scheme.integrate(f_advc_kk_j_check, triangle_vertices.T)
        #     quadpy_advc_kl_j_check = scheme.integrate(f_advc_kl_j_check, triangle_vertices.T)
        #     quadpy_advc_lk_j_check = scheme.integrate(f_advc_lk_j_check, triangle_vertices.T)
        #     quadpy_advc_ll_j_check = scheme.integrate(f_advc_ll_j_check, triangle_vertices.T)
        #     # --- LOOPING OVER QUADRATURE POINTS AND WEIGHTS IN CELL, AND UPDATING THE NUMERICAL INTEGRAL
        #     for _qc in range(len(cell_triangle.quadrature_weights)):
        #         _x_qc = cell_triangle.quadrature_points[:, _qc]
        #         _w_qc = cell_triangle.quadrature_weights[_qc]
        #         phi_k = cell_basis_k.get_phi_vector(_x_qc, x_c, h_c)
        #         phi_l = cell_basis_l.get_phi_vector(_x_qc, x_c, h_c)
        #         d_phi_k_i = cell_basis_k.get_d_phi_vector(_x_qc, x_c, h_c, _i)
        #         d_phi_k_j = cell_basis_k.get_d_phi_vector(_x_qc, x_c, h_c, _j)
        #         d_phi_l_i = cell_basis_l.get_d_phi_vector(_x_qc, x_c, h_c, _i)
        #         d_phi_l_j = cell_basis_l.get_d_phi_vector(_x_qc, x_c, h_c, _j)
        #         # ---
        #         phi_k_check += _w_qc * phi_k @ coefficients_k
        #         phi_l_check += _w_qc * phi_l @ coefficients_l
        #         # ---
        #         d_phi_k_i_check += _w_qc * d_phi_k_i @ coefficients_k
        #         d_phi_l_i_check += _w_qc * d_phi_l_i @ coefficients_l
        #         d_phi_k_j_check += _w_qc * d_phi_k_j @ coefficients_k
        #         d_phi_l_j_check += _w_qc * d_phi_l_j @ coefficients_l
        #         # ---
        #         mass_kk_check += _w_qc * coefficients_k @ np.tensordot(phi_k, phi_k, axes=0) @ coefficients_k
        #         mass_kl_check += _w_qc * coefficients_k @ np.tensordot(phi_k, phi_l, axes=0) @ coefficients_l
        #         mass_lk_check += _w_qc * coefficients_l @ np.tensordot(phi_l, phi_k, axes=0) @ coefficients_k
        #         mass_ll_check += _w_qc * coefficients_l @ np.tensordot(phi_l, phi_l, axes=0) @ coefficients_l
        #         # ---
        #         stif_kk_ii_check += (
        #             _w_qc * coefficients_k @ np.tensordot(d_phi_k_i, d_phi_k_i, axes=0) @ coefficients_k
        #         )
        #         stif_kl_ii_check += (
        #             _w_qc * coefficients_k @ np.tensordot(d_phi_k_i, d_phi_l_i, axes=0) @ coefficients_l
        #         )
        #         stif_lk_ii_check += (
        #             _w_qc * coefficients_l @ np.tensordot(d_phi_l_i, d_phi_k_i, axes=0) @ coefficients_k
        #         )
        #         stif_ll_ii_check += (
        #             _w_qc * coefficients_l @ np.tensordot(d_phi_l_i, d_phi_l_i, axes=0) @ coefficients_l
        #         )
        #         # ---
        #         stif_kk_ij_check += (
        #             _w_qc * coefficients_k @ np.tensordot(d_phi_k_i, d_phi_k_j, axes=0) @ coefficients_k
        #         )
        #         stif_kl_ij_check += (
        #             _w_qc * coefficients_k @ np.tensordot(d_phi_k_i, d_phi_l_j, axes=0) @ coefficients_l
        #         )
        #         stif_lk_ij_check += (
        #             _w_qc * coefficients_l @ np.tensordot(d_phi_l_i, d_phi_k_j, axes=0) @ coefficients_k
        #         )
        #         stif_ll_ij_check += (
        #             _w_qc * coefficients_l @ np.tensordot(d_phi_l_i, d_phi_l_j, axes=0) @ coefficients_l
        #         )
        #         # ---
        #         stif_kk_ji_check += (
        #             _w_qc * coefficients_k @ np.tensordot(d_phi_k_j, d_phi_k_i, axes=0) @ coefficients_k
        #         )
        #         stif_kl_ji_check += (
        #             _w_qc * coefficients_k @ np.tensordot(d_phi_k_j, d_phi_l_i, axes=0) @ coefficients_l
        #         )
        #         stif_lk_ji_check += (
        #             _w_qc * coefficients_l @ np.tensordot(d_phi_l_j, d_phi_k_i, axes=0) @ coefficients_k
        #         )
        #         stif_ll_ji_check += (
        #             _w_qc * coefficients_l @ np.tensordot(d_phi_l_j, d_phi_l_i, axes=0) @ coefficients_l
        #         )
        #         # ---
        #         stif_kk_jj_check += (
        #             _w_qc * coefficients_k @ np.tensordot(d_phi_k_j, d_phi_k_j, axes=0) @ coefficients_k
        #         )
        #         stif_kl_jj_check += (
        #             _w_qc * coefficients_k @ np.tensordot(d_phi_k_j, d_phi_l_j, axes=0) @ coefficients_l
        #         )
        #         stif_lk_jj_check += (
        #             _w_qc * coefficients_l @ np.tensordot(d_phi_l_j, d_phi_k_j, axes=0) @ coefficients_k
        #         )
        #         stif_ll_jj_check += (
        #             _w_qc * coefficients_l @ np.tensordot(d_phi_l_j, d_phi_l_j, axes=0) @ coefficients_l
        #         )
        #         # ---
        #         advc_kk_i_check += _w_qc * coefficients_k @ np.tensordot(phi_k, d_phi_k_i, axes=0) @ coefficients_k
        #         advc_kl_i_check += _w_qc * coefficients_k @ np.tensordot(phi_k, d_phi_l_i, axes=0) @ coefficients_l
        #         advc_lk_i_check += _w_qc * coefficients_l @ np.tensordot(phi_l, d_phi_k_i, axes=0) @ coefficients_k
        #         advc_ll_i_check += _w_qc * coefficients_l @ np.tensordot(phi_l, d_phi_l_i, axes=0) @ coefficients_l
        #         # ---
        #         advc_kk_j_check += _w_qc * coefficients_k @ np.tensordot(phi_k, d_phi_k_j, axes=0) @ coefficients_k
        #         advc_kl_j_check += _w_qc * coefficients_k @ np.tensordot(phi_k, d_phi_l_j, axes=0) @ coefficients_l
        #         advc_lk_j_check += _w_qc * coefficients_l @ np.tensordot(phi_l, d_phi_k_j, axes=0) @ coefficients_k
        #         advc_ll_j_check += _w_qc * coefficients_l @ np.tensordot(phi_l, d_phi_l_j, axes=0) @ coefficients_l
        #     # --- CHECKING AND COMPARING RESULTS
        #     rtol = 1.0e-15
        #     atol = 1.0e-15
        #     np.testing.assert_allclose(quadpy_phi_k_check, phi_k_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_phi_l_check, phi_l_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_d_phi_k_i_check, d_phi_k_i_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_d_phi_l_i_check, d_phi_l_i_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_d_phi_k_j_check, d_phi_k_j_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_d_phi_l_j_check, d_phi_l_j_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_mass_kk_check, mass_kk_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_mass_kl_check, mass_kl_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_mass_lk_check, mass_lk_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_mass_ll_check, mass_ll_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_kk_ii_check, stif_kk_ii_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_kl_ii_check, stif_kl_ii_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_lk_ii_check, stif_lk_ii_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_ll_ii_check, stif_ll_ii_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_kk_ij_check, stif_kk_ij_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_kl_ij_check, stif_kl_ij_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_lk_ij_check, stif_lk_ij_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_ll_ij_check, stif_ll_ij_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_kk_ji_check, stif_kk_ji_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_kl_ji_check, stif_kl_ji_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_lk_ji_check, stif_lk_ji_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_ll_ji_check, stif_ll_ji_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_kk_jj_check, stif_kk_jj_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_kl_jj_check, stif_kl_jj_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_lk_jj_check, stif_lk_jj_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_stif_ll_jj_check, stif_ll_jj_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_advc_kk_i_check, advc_kk_i_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_advc_kl_i_check, advc_kl_i_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_advc_lk_i_check, advc_lk_i_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_advc_ll_i_check, advc_ll_i_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_advc_kk_j_check, advc_kk_j_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_advc_kl_j_check, advc_kl_j_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_advc_lk_j_check, advc_lk_j_check, rtol=rtol, atol=atol)
        #     np.testing.assert_allclose(quadpy_advc_ll_j_check, advc_ll_j_check, rtol=rtol, atol=atol)
        return

    def test_element_triangle(self):

        return
        #                 # --------------------------------------------------------------------------------------------------------------
        # # COMPUTE PHI VECTOR INTEGRAL
        # # --------------------------------------------------------------------------------------------------------------
        # # --- COMPUTE INTEGRAL
        # orders = [face_polynomial_order, cell_polynomial_order]
        # bases = [cell_basis_k, cell_basis_l]
        # coefficients = [coefficients_k, coefficients_l]
        # labels = ["FACE", "CELL"]
        # for label, order, cell_basis, coefficients in zip(labels, orders, bases, coefficients):
        #     computed_val = 0.0
        #     for _qc in range(len(cell_triangle.quadrature_weights)):
        #         _x_qc = cell_triangle.quadrature_points[:, _qc]
        #         _w_qc = cell_triangle.quadrature_weights[_qc]
        #         phi0 = cell_basis.get_phi_vector(_x_qc, x_c, h_c)
        #         computed_val += _w_qc * phi0 @ coefficients
        #     # for item in quadpy.t2.schemes.items():
        #     #     print(item)
        #     # --- GET QUADPY INTEGRAL EVALUATION
        #     scheme = quadpy.t2.get_good_scheme(order)
        #     f = lambda x: test_function(x, x_c, h_c, coefficients)
        #     val = scheme.integrate(f, triangle_vertices.T)
        #     print("COMPUTE PHI VECTOR INTEGRAL -- {} ORDER {}".format(label, order))
        #     print("num_res : {}".format(computed_val))
        #     print("val_res : {}".format(val))
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # COMPUTE DERIVATIVE PHI VECTOR INTEGRAL
        # # --------------------------------------------------------------------------------------------------------------
        # # --- COMPUTE INTEGRAL
        # orders = [face_polynomial_order, cell_polynomial_order]
        # bases = [cell_basis_k, cell_basis_l]
        # coefficients = [coefficients_k, coefficients_l]
        # labels = ["FACE", "CELL"]
        # for label, order, cell_basis, coefficients in zip(labels, orders, bases, coefficients):
        #     for _i in range(euclidean_dimension):
        #         computed_val = 0.0
        #         for _qc in range(len(cell_triangle.quadrature_weights)):
        #             _x_qc = cell_triangle.quadrature_points[:, _qc]
        #             _w_qc = cell_triangle.quadrature_weights[_qc]
        #             phi0 = cell_basis.get_d_phi_vector(_x_qc, x_c, h_c, _i)
        #             computed_val += _w_qc * phi0 @ coefficients
        #         # for item in quadpy.t2.schemes.items():
        #         #     print(item)
        #         # --- GET QUADPY INTEGRAL EVALUATION
        #         scheme = quadpy.t2.get_good_scheme(order)
        #         f = lambda x: test_function_derivative(x, x_c, h_c, _i, coefficients)
        #         val = scheme.integrate(f, triangle_vertices.T)
        #         print("COMPUTE PHI VECTOR INTEGRAL -- {} ORDER {} -- DIRECTION {}".format(label, order, _i))
        #         print("num_res : {}".format(computed_val))
        #         print("val_res : {}".format(val))
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # COMPUTE MASS MATRIX INTEGRAL
        # # --------------------------------------------------------------------------------------------------------------
        # # --- COMPUTE INTEGRAL
        # orders = [face_polynomial_order, cell_polynomial_order]
        # bases = [cell_basis_k, cell_basis_l]
        # coefficients = [coefficients_k, coefficients_l]
        # labels = ["FACE", "CELL"]
        # for label, order, cell_basis, coefficients in zip(labels, orders, bases, coefficients):
        #     computed_val = 0.0
        #     for _qc in range(len(cell_triangle.quadrature_weights)):
        #         _x_qc = cell_triangle.quadrature_points[:, _qc]
        #         _w_qc = cell_triangle.quadrature_weights[_qc]
        #         phi0 = cell_basis.get_phi_vector(_x_qc, x_c, h_c)
        #         phi1 = cell_basis.get_phi_vector(_x_qc, x_c, h_c)
        #         mat = np.tensordot(phi0, phi1, axes=0)
        #         computed_val += _w_qc * coefficients @ mat @ coefficients
        #     # for item in quadpy.t2.schemes.items():
        #     #     print(item)
        #     # --- GET QUADPY INTEGRAL EVALUATION
        #     scheme = quadpy.t2.get_good_scheme(2 * order)
        #     f = lambda x: test_function(x, x_c, h_c, coefficients) ** 2
        #     val = scheme.integrate(f, triangle_vertices.T)
        #     print("COMPUTE PHI VECTOR INTEGRAL -- {} ORDER {}".format(label, order))
        #     print("num_res : {}".format(computed_val))
        #     print("val_res : {}".format(val))
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # COMPUTE STIFFNESS MATRIX INTEGRAL
        # # --------------------------------------------------------------------------------------------------------------
        # # --- COMPUTE INTEGRAL
        # orders = [face_polynomial_order, cell_polynomial_order]
        # bases = [cell_basis_k, cell_basis_l]
        # coefficients = [coefficients_k, coefficients_l]
        # labels = ["FACE", "CELL"]
        # for label, order, cell_basis, coefficients in zip(labels, orders, bases, coefficients):
        #     for _i in range(euclidean_dimension):
        #         for _j in range(euclidean_dimension):
        #             computed_val = 0.0
        #             for _qc in range(len(cell_triangle.quadrature_weights)):
        #                 _x_qc = cell_triangle.quadrature_points[:, _qc]
        #                 _w_qc = cell_triangle.quadrature_weights[_qc]
        #                 phi0 = cell_basis.get_d_phi_vector(_x_qc, x_c, h_c, _i)
        #                 phi1 = cell_basis.get_d_phi_vector(_x_qc, x_c, h_c, _j)
        #                 mat = np.tensordot(phi0, phi1, axes=0)
        #                 computed_val += _w_qc * coefficients @ mat @ coefficients
        #             # for item in quadpy.t2.schemes.items():
        #             #     print(item)
        #             # --- GET QUADPY INTEGRAL EVALUATION
        #             scheme = quadpy.t2.get_good_scheme(2 * order)
        #             f = lambda x: test_function_derivative(x, x_c, h_c, _i, coefficients) * test_function_derivative(x, x_c, h_c, _j, coefficients)
        #             val = scheme.integrate(f, triangle_vertices.T)
        #             print("COMPUTE PHI VECTOR INTEGRAL -- {} ORDER {} -- DIRECTIONs {} {}".format(label, order, _i, _j))
        #             print("num_res : {}".format(computed_val))
        #             print("val_res : {}".format(val))
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # COMPUTE DERIVATIVE PHI VECTOR INTEGRAL
        # # --------------------------------------------------------------------------------------------------------------
        # # --- COMPUTE INTEGRAL
        # for _i in range(euclidean_dimension):
        #     computed_val = 0.0
        #     for _qc in range(len(cell_triangle.quadrature_weights)):
        #         _x_qc = cell_triangle.quadrature_points[:, _qc]
        #         _w_qc = cell_triangle.quadrature_weights[_qc]
        #         computed_val += _w_qc * cell_basis_k.get_d_phi_vector(_x_qc, x_c, h_c, _i) @ coefficients_k
        #     # for item in quadpy.t2.schemes.items():
        #     #     print(item)
        #     # --- GET QUADPY INTEGRAL EVALUATION
        #     scheme = quadpy.t2.get_good_scheme(face_polynomial_order)
        #     f = lambda x: test_function_derivative(x, x_c, h_c, _i, coefficients_k)
        #     val = scheme.integrate(f, triangle_vertices.T)
        #     print("COMPUTE DERIVATIVE PHI VECTOR INTEGRAL --- DIRECTION {}".format(_i))
        #     print("num_res : {}".format(computed_val))
        #     print("val_res : {}".format(val))
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # COMPUTE STIFFNESS MATRIX INTEGRAL
        # # --------------------------------------------------------------------------------------------------------------
        # # --- COMPUTE INTEGRAL
        # for _i in range(euclidean_dimension):
        #     computed_val = 0.0
        #     for _qc in range(len(cell_triangle.quadrature_weights)):
        #         _x_qc = cell_triangle.quadrature_points[:, _qc]
        #         _w_qc = cell_triangle.quadrature_weights[_qc]
        #         phi0 = cell_basis_k.get_d_phi_vector(_x_qc, x_c, h_c, _i)
        #         phi1 = cell_basis_k.get_d_phi_vector(_x_qc, x_c, h_c, _i)
        #         mat = np.tensordot(phi0, phi1, axes=0)
        #         computed_val += _w_qc * coefficients_k @ mat @ coefficients_k
        #     # --- GET QUADPY INTEGRAL EVALUATION
        #     scheme = quadpy.t2.get_good_scheme(integration_order)
        #     f = lambda x : test_function_derivative(x, x_c, h_c, _i, coefficients_k) **2
        #     val = scheme.integrate(f, triangle_vertices.T)
        #     print("COMPUTE STIFFNESS MATRIX INTEGRAL --- DIRECTION {}".format(_i))
        #     print("num_res : {}".format(computed_val))
        #     print("val_res : {}".format(val))
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # COMPUTE ADVECTION MATRIX INTEGRAL
        # # --------------------------------------------------------------------------------------------------------------
        # # --- COMPUTE INTEGRAL
        # for _i in range(euclidean_dimension):
        #     computed_val = 0.0
        #     for _qc in range(len(cell_triangle.quadrature_weights)):
        #         _x_qc = cell_triangle.quadrature_points[:, _qc]
        #         _w_qc = cell_triangle.quadrature_weights[_qc]
        #         phi0 = cell_basis_k.get_phi_vector(_x_qc, x_c, h_c)
        #         phi1 = cell_basis_k.get_d_phi_vector(_x_qc, x_c, h_c, _i)
        #         mat = np.tensordot(phi0, phi1, axes=0)
        #         computed_val += _w_qc * coefficients_k @ mat @ coefficients_k
        #     # --- GET QUADPY INTEGRAL EVALUATION
        #     scheme = quadpy.t2.get_good_scheme(integration_order)
        #     f = lambda x: test_function_derivative(x, x_c, h_c, _i, coefficients_k) * test_function(x, x_c, h_c, coefficients_k)
        #     val = scheme.integrate(f, triangle_vertices.T)
        #     print("COMPUTE ADVECTION MATRIX INTEGRAL --- DIRECTION {}".format(_i))
        #     print("num_res : {}".format(computed_val))
        #     print("val_res : {}".format(val))
