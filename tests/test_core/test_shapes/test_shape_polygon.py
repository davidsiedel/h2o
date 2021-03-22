from unittest import TestCase
from h2o.h2o import *

import h2o.geometry.shapes.shape_polygon as shape_polygon


class TestShapePolygon(TestCase):
    def test_polygon_centroid(self):
        v0 = [0.0, 0.0]
        v01 = [0.5, 0.0]
        v1 = [1.0, 0.0]
        v2 = [1.0, 1.0]
        v3 = [0.5, 1.0]
        v4 = [0.0, 1.0]
        polygon_vertices = np.array([
            v0,
            v01,
            v1,
            v2,
            v3,
            v4,
        ]).T
        polygon_partition = shape_polygon.get_polygon_partition(polygon_vertices)
        print(polygon_partition)
        polygon_rotation = shape_polygon.get_polygon_rotation_matrix(polygon_vertices)
        print(polygon_rotation)
        polygon_volume = shape_polygon.get_polygon_volume(polygon_vertices)
        print(polygon_volume)
        self.assertTrue(True)