from h2o.geometry.shape import Shape, get_rotation_matrix
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
    "FieldType": FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN,
}

class TriangleTest:
    def __init__(
        self,
        field: Field,
        finite_element: FiniteElement,
        vertices: np.ndarray
    ) -> None:
        TriangleTest.check_vertices(vertices)
        self.field = field
        self.cell: Shape = Shape(ShapeType.TRIANGLE, vertices)
        self.faces: List[Shape] = [
            Shape(ShapeType.SEGMENT, vertices[:, [0, 1]]),
            Shape(ShapeType.SEGMENT, vertices[:, [1, 2]]),
            Shape(ShapeType.SEGMENT, vertices[:, [2, 0]]),
        ]
        self.finite_element = finite_element
        self.gradients: List[np.ndarray] = grad.get_gradient_operators(field, finite_element, self.cell, self.faces)
        self.stabilization: np.ndarray = stab.get_stabilization_operator(field, finite_element, self.cell, self.faces)

    def get_symmetric_cartesian_gradient_component_mat(
            field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], _i: int, _j: int
    ) -> ndarray:
        _d = field.euclidean_dimension
        _dx = field.field_dimension
        _cl = finite_element.cell_basis_l.dimension
        _ck = finite_element.cell_basis_k.dimension
        _fk = finite_element.face_basis_k.dimension
        _nf = len(faces)
        _es = _dx * (_cl + _nf * _fk)
        local_grad_matric = np.zeros((_ck, _es), dtype=real)
        x_c = cell.get_centroid()
        h_c = cell.get_diameter()
        bdc = cell.get_bounding_box()
        _io = finite_element.construction_integration_order
        print("io :", _io)
        _c_is = cell.get_quadrature_size(_io)
        cell_quadrature_points = cell.get_quadrature_points(_io)
        cell_quadrature_weights = cell.get_quadrature_weights(_io)
        m_mas = np.zeros((_ck, _ck), dtype=real)
        for qc in range(_c_is):
            x_q_c = cell_quadrature_points[:, qc]
            w_q_c = cell_quadrature_weights[qc]
            phi_k = finite_element.cell_basis_k.evaluate_function(x_q_c, x_c, bdc)
            d_phi_l_j = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, bdc, _j)
            d_phi_l_i = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, bdc, _i)
            m_mas += w_q_c * np.tensordot(phi_k, phi_k, axes=0)
            _c0 = _i * _cl
            _c1 = (_i + 1) * _cl
            local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * w_q_c * np.tensordot(phi_k, d_phi_l_j, axes=0)
            _c0 = _j * _cl
            _c1 = (_j + 1) * _cl
            local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * w_q_c * np.tensordot(phi_k, d_phi_l_i, axes=0)
        m_mas_inv = np.linalg.inv(m_mas)
        print("local_grad_matric after cell ", _i, _j)
        print(local_grad_matric)
        for _f, face in enumerate(faces):
            print("--------- seg ", _f)
            # --- FACE GEOMETRY
            x_f = face.get_centroid()
            bdf_proj = face.get_face_bounding_box()
            face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
            dist_in_face = (face_rotation_matrix @ (x_f - x_c))[-1]
            if dist_in_face > 0:
                normal_vector_component_j = face_rotation_matrix[-1, _j]
                normal_vector_component_i = face_rotation_matrix[-1, _i]
            else:
                normal_vector_component_j = -face_rotation_matrix[-1, _j]
                normal_vector_component_i = -face_rotation_matrix[-1, _i]
            _io = finite_element.construction_integration_order
            _f_is = face.get_quadrature_size(_io)
            face_quadrature_points = face.get_quadrature_points(_io)
            face_quadrature_weights = face.get_quadrature_weights(_io)
            for qf in range(_f_is):
                x_q_f = face_quadrature_points[:, qf]
                w_q_f = face_quadrature_weights[qf]
                s_f = (face_rotation_matrix @ x_f)[:-1]
                s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
                phi_k = finite_element.cell_basis_k.evaluate_function(x_q_f, x_c, bdc)
                phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
                psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
                print("*** phi_l ", qf)
                print(phi_l)
                print("*** phi_k ", qf)
                print(phi_k)
                print("*** psi_k ", qf)
                print(psi_k)
                _c0 = _i * _cl
                _c1 = (_i + 1) * _cl
                local_grad_matric[:, _c0:_c1] -= (1.0 / 2.0) * w_q_f * np.tensordot(phi_k, phi_l,
                                                                                    axes=0) * normal_vector_component_j
                _c0 = _dx * _cl + _f * _dx * _fk + _i * _fk
                _c1 = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
                local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * w_q_f * np.tensordot(phi_k, psi_k,
                                                                                    axes=0) * normal_vector_component_j
                _c0 = _j * _cl
                _c1 = (_j + 1) * _cl
                local_grad_matric[:, _c0:_c1] -= (1.0 / 2.0) * w_q_f * np.tensordot(phi_k, phi_l,
                                                                                    axes=0) * normal_vector_component_i
                _c0 = _dx * _cl + _f * _dx * _fk + _j * _fk
                _c1 = _dx * _cl + _f * _dx * _fk + (_j + 1) * _fk
                local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * w_q_f * np.tensordot(phi_k, psi_k,
                                                                                    axes=0) * normal_vector_component_i
        print("local_grad_matric after face ", _i, _j)
        print(local_grad_matric)
        local_grad_matric2 = m_mas_inv @ local_grad_matric
        return local_grad_matric2


    def get_regular_cartesian_gradient_component_mat(
            field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], _i: int, _j: int
    ) -> ndarray:
        _d = field.euclidean_dimension
        _dx = field.field_dimension
        _cl = finite_element.cell_basis_l.dimension
        _ck = finite_element.cell_basis_k.dimension
        _fk = finite_element.face_basis_k.dimension
        _nf = len(faces)
        _es = _dx * (_cl + _nf * _fk)
        local_grad_matric = np.zeros((_ck, _es), dtype=real)
        x_c = cell.get_centroid()
        bdc = cell.get_bounding_box()
        _io = finite_element.construction_integration_order
        _c_is = cell.get_quadrature_size(_io)
        cell_quadrature_points = cell.get_quadrature_points(_io)
        cell_quadrature_weights = cell.get_quadrature_weights(_io)
        m_mas = np.zeros((_ck, _ck), dtype=real)
        for qc in range(_c_is):
            x_q_c = cell_quadrature_points[:, qc]
            w_q_c = cell_quadrature_weights[qc]
            phi_k = finite_element.cell_basis_k.evaluate_function(x_q_c, x_c, bdc)
            d_phi_l_j = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, bdc, _j)
            m_mas += w_q_c * np.tensordot(phi_k, phi_k, axes=0)
            _c0 = _i * _cl
            _c1 = (_i + 1) * _cl
            local_grad_matric[:, _c0:_c1] += w_q_c * np.tensordot(phi_k, d_phi_l_j, axes=0)
        m_mas_inv = np.linalg.inv(m_mas)
        for _f, face in enumerate(faces):
            # --- FACE GEOMETRY
            x_f = face.get_centroid()
            bdf_proj = face.get_face_bounding_box()
            face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
            dist_in_face = (face_rotation_matrix @ (x_f - x_c))[-1]
            if dist_in_face > 0:
                normal_vector_component_j = face_rotation_matrix[-1, _j]
            else:
                normal_vector_component_j = -face_rotation_matrix[-1, _j]
            _io = finite_element.construction_integration_order
            _f_is = face.get_quadrature_size(_io)
            face_quadrature_points = face.get_quadrature_points(_io)
            face_quadrature_weights = face.get_quadrature_weights(_io)
            for qf in range(_f_is):
                x_q_f = face_quadrature_points[:, qf]
                w_q_f = face_quadrature_weights[qf]
                s_f = (face_rotation_matrix @ x_f)[:-1]
                s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
                phi_k = finite_element.cell_basis_k.evaluate_function(x_q_f, x_c, bdc)
                phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
                psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
                _c0 = _i * _cl
                _c1 = (_i + 1) * _cl
                local_grad_matric[:, _c0:_c1] -= w_q_f * np.tensordot(phi_k, phi_l, axes=0) * normal_vector_component_j
                _c0 = _dx * _cl + _f * _dx * _fk + _i * _fk
                _c1 = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
                local_grad_matric[:, _c0:_c1] += w_q_f * np.tensordot(phi_k, psi_k, axes=0) * normal_vector_component_j
        local_grad_matric2 = m_mas_inv @ local_grad_matric
        return local_grad_matric2

    def get_symmetric_cartesian_gradient_component_matrix(self, _i: int, _j: int):
        return TriangleTest.get_symmetric_cartesian_gradient_component_mat(self.field, self.finite_element, self.cell, self.faces, _i, _j)

    def get_regular_cartesian_gradient_component_matrix(self, _i: int, _j: int):
        return TriangleTest.get_regular_cartesian_gradient_component_mat(self.field, self.finite_element, self.cell, self.faces, _i, _j)
    
    def check_vertices(
        vertices: np.ndarray
    ) -> None:
        if vertices.shape != (2, 3):
            raise ValueError("Vertices do not match these of a Triangle and must be of shape (2, 3)")

    def print_gradients(
        self
    ) -> None:
        for i, gradient in enumerate(self.gradients):
            print("gradient matrix ", i)
            print(gradient)

    def print_gradient(
        self, _i: int, _j: int
    ) -> None:
        print(self.get_symmetric_cartesian_gradient_component_matrix(_i, _j))
        # for i, gradient in enumerate(self.gradients):
        #     print("gradient matrix ", i)
        #     print(gradient)

    def print_stabilization(
        self
    ) -> None:
        print(self.stabilization)

    def print_seg_mass(
        self
    ) -> None:
        x_c = self.cell.get_centroid()
        bdc = self.cell.get_bounding_box()
        _io = self.finite_element.construction_integration_order
        print("io : ", _io)
        for _f, face in enumerate(self.faces):
            print("--- seg ", _f)
            mass = np.zeros((2, 2))
            # --- FACE GEOMETRY
            x_f = face.get_centroid()
            bdf_proj = face.get_face_bounding_box()
            face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
            dist_in_face = (face_rotation_matrix @ (x_f - x_c))[-1]
            # if dist_in_face > 0:
            #     normal_vector_component_j = face_rotation_matrix[-1, _j]
            #     normal_vector_component_i = face_rotation_matrix[-1, _i]
            # else:
            #     normal_vector_component_j = -face_rotation_matrix[-1, _j]
            #     normal_vector_component_i = -face_rotation_matrix[-1, _i]
            _f_is = face.get_quadrature_size(_io)
            face_quadrature_points = face.get_quadrature_points(_io)
            face_quadrature_weights = face.get_quadrature_weights(_io)
            for qf in range(_f_is):
                x_q_f = face_quadrature_points[:, qf]
                w_q_f = face_quadrature_weights[qf]
                s_f = (face_rotation_matrix @ x_f)[:-1]
                s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
                # print("wt ", qf)
                # print(w_q_f)
                # print("pt ", qf)
                # print(x_q_f)
                # print("n ", qf)
                # print(face_rotation_matrix[-1, :])
                # print("s_q_f ", qf)
                # print(s_q_f)
                # print("s_f ", qf)
                # print(s_f)
                # print("dist ", qf)
                # print(s_q_f - s_f)
                # print("diams ", qf)
                # print(bdf_proj)
                # print("rot ", qf)
                # print(face_rotation_matrix)
                # phi_k = self.finite_element.cell_basis_k.evaluate_function(x_q_f, x_c, bdc)
                phi_l = self.finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
                psi_k = self.finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
                print("phi_l ", qf)
                print(phi_l)
                print("psi_k ", qf)
                print(psi_k)
                mass +=  w_q_f * np.tensordot(psi_k, psi_k, axes=0)
            # print("mass ")
            # print(mass)


# v0 = np.array([1.0, 1.7], dtype=real)
# v1 = np.array([2.0, 1.6], dtype=real)
# v2 = np.array([1.9, 3.0], dtype=real)
v0 = np.array([0.1, 0.0], dtype=real)
v1 = np.array([1.2, 0.0], dtype=real)
v2 = np.array([0.0, 0.9], dtype=real)
triangle_vertices = np.array([v0, v1, v2], dtype=real).T

triangle = TriangleTest(
    Field(
        "TEST",
        parameters["FieldType"]
    ),
    FiniteElement(
        element_type=parameters["ElementType"],
        polynomial_order=parameters["PolynomialOrder"],
        euclidean_dimension=parameters["EuclideanDimension"],
    ),
    triangle_vertices
)

triangle.print_gradients()
# triangle.print_gradient(0, 1)
triangle.print_stabilization()
# triangle.print_seg_mass()