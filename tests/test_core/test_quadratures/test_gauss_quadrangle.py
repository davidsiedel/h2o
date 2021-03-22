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
    def test_gauss_quadrangle(self):
        v0 = [0.02, 0.0]
        v1 = [1.0, 0.0]
        v2 = [1.0, 1.2]
        v3 = [0.0, 1.0]
        quadrangle_vertices = np.array([
            v0, v1, v2, v3
        ]).T
        cell = Shape(ShapeType.QUADRANGLE, quadrangle_vertices)
        a = np.sqrt(1.**2 + 2.**2)
        for _io in range(1, 9):
            print("---")
            scheme_io = _io
            scheme = quadpy.c2.get_good_scheme(scheme_io)
            # f = lambda x: np.exp(x[0])
            f = lambda x: np.exp(x[0]) * np.sin(x[0] * x[1]) + x[1]
            val = scheme.integrate(
                f,
                # [[[0.0, 0.0], [2.0, 0.0]], [[0.0, 1.0], [2.0, 1.0]]],
                # [[[0.0, 0.0], [2., 0.2]], [[0., 1.3], [2.1, 1.0]]],
                # [[[0., 0.], [1., 0.]], [[0., 1.], [1., 1.]]],
                [[v0, v1], [v3, v2]]
                # quadrangle_vertices.T,
            )
            val_num = 0.0
            cell_quadrature_points = cell.get_quadrature_points(_io)
            cell_quadrature_weights = cell.get_quadrature_weights(_io)
            cell_quadrature_size = cell.get_quadrature_size(_io)
            for _qc in range(cell_quadrature_size):
                x_qc = cell_quadrature_points[:,_qc]
                w_qc = cell_quadrature_weights[_qc]
                val_num += w_qc * f(x_qc)
            print("--- order {}".format(_io))
            print("val_num : {}".format(val_num))
            print("val_qud : {}".format(val))
        self.assertTrue(True)