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


class TestFaceSegment(TestCase):
    def test_face_segment(self, verbose=False):

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
                if verbose:
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
                    if verbose:
                        print("QUAD_POINT : {} | QUAD_WEIGHT : {}".format(x_qf, w_qf))
                # --- SET PLOT OPTIONS
                plt.gca().set_aspect("equal", adjustable="box")
                plt.grid()
                plt.show()

                # --------------------------------------------------------------------------------------------------------------
                # CHECK DISTANCES
                # --------------------------------------------------------------------------------------------------------------
                if verbose:
                    print("DIAMETER : {}".format(h_f))
                    print("DIST ORIGINAL : {}".format(np.linalg.norm(v_0 - v_1)))
                    print("DIST PROJECTION : {}".format(np.linalg.norm(s_0 - s_1)))

                # --------------------------------------------------------------------------------------------------------------
                # CHECK INTEGRATION IN FACE
                # --------------------------------------------------------------------------------------------------------------
                # scheme = quadpy.c1.gauss_legendre(2 * _io)
                scheme = quadpy.c1.gauss_legendre(_io)
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
                        print('MASS INTEGRAL CHECK | ORDER : {} | ELEM : {}'.format(polynomial_order, element_type))
                        print("- QUADPY : {}".format(mass_integral_check))
                        print("- H2O : {}".format(mass_integral))
                        print(
                            'STIFFNESS INTEGRAL CHECK | ORDER : {} | ELEM : {}'.format(polynomial_order, element_type))
                        print("- QUADPY : {}".format(stif_integral_check))
                        print("- H2O : {}".format(stif_integral))
                        print(
                            'ADVECTION INTEGRAL CHECK | ORDER : {} | ELEM : {}'.format(polynomial_order, element_type))
                        print("- QUADPY : {}".format(advc_integral_check))
                        print("- H2O : {}".format(advc_integral))
                        np.testing.assert_allclose(mass_integral_check, mass_integral, rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stif_integral_check, stif_integral, rtol=rtol, atol=atol)
                        np.testing.assert_allclose(advc_integral_check, advc_integral, rtol=rtol, atol=atol)