from h2o.geometry.shape import Shape
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
import h2o.fem.element.operators.gradient as grad
import h2o.fem.element.operators.stabilization as stab
from h2o.h2o import *

np.set_printoptions(precision=4)

parameters = {
    "ElementType": ElementType.HHO_EQUAL,
    "PolynomialOrder": 1,
    "EuclideanDimension": 2,
    "FieldType": FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN,
}

class QuadrangleTest:
    def __init__(
        self,
        field: Field,
        finite_element: FiniteElement,
        vertices: np.ndarray
    ) -> None:
        cell: Shape = Shape(ShapeType.QUADRANGLE, vertices)
        faces: List[Shape] = [
            Shape(ShapeType.SEGMENT, vertices[:, [0, 1]]),
            Shape(ShapeType.SEGMENT, vertices[:, [1, 2]]),
            Shape(ShapeType.SEGMENT, vertices[:, [2, 3]]),
            Shape(ShapeType.SEGMENT, vertices[:, [3, 0]]),
        ]
        self.gradients: List[np.ndarray] = grad.get_gradient_operators(field, finite_element, cell, faces)
        self.stabilization: np.ndarray = stab.get_stabilization_operator(field, finite_element, cell, faces)

    def check_vertices(
        vertices: np.ndarray
    ) -> None:
        if vertices.shape != (2, 4):
            raise ValueError("Vertices do not match these of a Triangle and must be of shape (2, 4)")

    def print_gradients(
        self
    ) -> None:
        for i, gradient in enumerate(self.gradients):
            np.set_printoptions(precision=3)
            print("gradient matrix ", i)
            # print(gradient)
            print(mat2str(gradient, 3))

    def print_stabilization(
        self
    ) -> None:
        np.set_printoptions(precision=3)
        print("stabilization")
        # print(self.stabilization)
        print(mat2str(self.stabilization, 3))
        print(mat2str(self.stabilization[:, 1], 3))


v0 = np.array([0.0, 0.0], dtype=real)
v1 = np.array([1.0, 0.0], dtype=real)
v2 = np.array([1.0, 1.0], dtype=real)
v3 = np.array([0.0, 1.0], dtype=real)
quadrangle_vertices = np.array([v0, v1, v2, v3], dtype=real).T

quadrangle = QuadrangleTest(
    Field(
        "TEST",
        parameters["FieldType"]
    ),
    FiniteElement(
        element_type=parameters["ElementType"],
        polynomial_order=parameters["PolynomialOrder"],
        euclidean_dimension=parameters["EuclideanDimension"],
    ),
    quadrangle_vertices
)

quadrangle.print_gradients()
quadrangle.print_stabilization()