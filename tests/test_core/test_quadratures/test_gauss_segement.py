from random import uniform
from unittest import TestCase

import matplotlib.pyplot as plt
import quadpy

from h2o.fem.basis.bases.monomial import Monomial
from h2o.geometry.shape import Shape, get_rotation_matrix
from h2o.h2o import *

np.set_printoptions(precision=16)
np.set_printoptions(linewidth=1)


class TestElementTriangle(TestCase):
    def test_gauss_segment(self):
        v0 = [0.0, 0.0]
        v1 = [1.0, 0.0]
        quadrangle_vertices = np.array([
            v0, v1
        ]).T
        cell = Shape(ShapeType.SEGMENT, quadrangle_vertices)
        for _io in range(1, 9):
            print("---")
            # f = lambda x: np.exp(x[0])
            f = lambda x: np.exp(x) * np.sin(x) + x
            seg = quadpy.c1.gauss_legendre(_io)
            val = seg.integrate(f, [v0[0], v1[0]])
            # val = quadpy.quad(f, v0[0], v1[0])
            val_num = 0.0
            cell_quadrature_points = cell.get_quadrature_points(_io)
            cell_quadrature_weights = cell.get_quadrature_weights(_io)
            cell_quadrature_size = cell.get_quadrature_size(_io)
            for _qc in range(cell_quadrature_size):
                x_qc = cell_quadrature_points[:,_qc]
                w_qc = cell_quadrature_weights[_qc]
                val_num += w_qc * f(x_qc)
            print("val_num : {}".format(val_num[0]))
            print("val_qud : {}".format(val))
            rtol = 1.e-11
            atol = 1.e-11
            np.testing.assert_allclose(val, val_num[0], rtol, atol)
        self.assertTrue(True)