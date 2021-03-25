from h2o.h2o import *
from h2o.geometry.shape import Shape
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
import h2o.fem.element.operators.operator_gradient as gradop
import h2o.fem.element.operators.operator_stabilization as stabop


class Element:
    cell: Shape
    faces: List[Shape]
    faces_indices: List[int]
    field: Field
    finite_element: FiniteElement
    element_size: int
    gradients_operators: ndarray
    stabilization_operator: ndarray
    cell_unknown_vector: ndarray

    def __init__(
        self, field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], faces_indices: List[int]
    ):
        """
        Args:
            field:
            finite_element:
            cell:
            faces:
            faces_indices:
        """
        self.cell = cell
        self.faces = faces
        self.faces_indices = faces_indices
        self.field = field
        self.finite_element = finite_element
        # --- GET INDICES
        _dx = field.field_dimension
        _cl = finite_element.cell_basis_l.dimension
        _fk = finite_element.face_basis_k.dimension
        _nf = len(self.faces)
        # --- BUILD LOCAL MATRICES AND VECTORS
        self.element_size = _dx * (_cl + _nf * _fk)
        self.m_cell_cell_inv = np.zeros((_cl * _dx, _cl * _dx), dtype=real)
        self.m_cell_faces = np.zeros((_cl * _dx, _nf * _fk * _dx), dtype=real)
        self.v_cell = np.zeros((_cl * _dx,), dtype=real)
        self.cell_unknown_vector = np.zeros((_dx * _cl,), dtype=real)
        # --- BUILD OPERATORS
        if finite_element.element_type in [ElementType.HDG_LOW, ElementType.HDG_EQUAL, ElementType.HDG_HIGH]:
            self.gradients_operators = gradop.get_gradient_operators(field, finite_element, cell, faces)
            # self.stabilization_operator = stabop.get_stabilization_operator(field, finite_element, cell, faces)
            self.stabilization_operator = stabop.get_stab_test(field, finite_element, cell, faces)
        else:
            raise KeyError("NO")
        return

    def get_element_unknown_vector(self, faces_global_unknown_vector: ndarray) -> ndarray:
        """

        Args:
            faces_global_unknown_vector:

        Returns:

        """
        _dx = self.field.field_dimension
        _fk = self.finite_element.face_basis_k.dimension
        _cl = self.finite_element.cell_basis_l.dimension
        element_unknown_vector = np.zeros((self.element_size,), dtype=real)
        for _f_local, _f_global in enumerate(self.faces_indices):
            _ci_g = _f_global * (_fk * _dx)
            _cj_g = (_f_global + 1) * (_fk * _dx)
            _ci_l = (_dx * _cl) + _f_local * (_fk * _dx)
            _cj_l = (_dx * _cl) + (_f_local + 1) * (_fk * _dx)
            element_unknown_vector[_ci_l:_cj_l] += faces_global_unknown_vector[_ci_g:_cj_g]
        _ci = _cl * _dx
        element_unknown_vector[:_ci] += self.cell_unknown_vector
        return element_unknown_vector

    def get_transformation_gradient(self, faces_unknown_vector: ndarray, _qc: int):
        """

        Args:
            faces_unknown_vector:
            _qc:

        Returns:

        """
        element_unknown_vector = self.get_element_unknown_vector(faces_unknown_vector)
        if self.field.grad_type == GradType.DISPLACEMENT_TRANSFORMATION_GRADIENT:
            ones_vect = np.zeros((self.field.gradient_dimension,), dtype=real)
            for indx in range(3):
                ones_vect[indx] += 1.0
            transformation_gradient_0 = self.gradients_operators[_qc] @ element_unknown_vector
            transformation_gradient = ones_vect + transformation_gradient_0
        elif self.field.grad_type == GradType.DISPLACEMENT_SMALL_STRAIN:
            transformation_gradient_0 = self.gradients_operators[_qc] @ element_unknown_vector
            transformation_gradient = transformation_gradient_0
        else:
            raise KeyError("No such strain type")
        return transformation_gradient

    def get_cell_field_value(self, faces_unknown_vector: ndarray, point: ndarray, direction: int) -> float:
        """

        Args:
            faces_unknown_vector:
            point:
            direction:

        Returns:

        """
        element_unknown_vector = self.get_element_unknown_vector(faces_unknown_vector)
        _cl = self.finite_element.cell_basis_l.dimension
        _c0 = direction * _cl
        _c1 = (direction + 1) * _cl
        x_c = self.cell.get_centroid()
        h_c = self.cell.get_diameter()
        vcl = self.finite_element.cell_basis_l.evaluate_function(point, x_c, h_c)
        field_unknown_value = vcl @ element_unknown_vector[_c0:_c1]
        return field_unknown_value


# from numpy import ndarray
# from typing import List, Dict, Tuple, Any
# from h2o.h2o import *
# # from h2o.pbbb.boundary_condition import BoundaryCondition
# # from h2o.pbbb.load import Load
# from h2o.fem.element.cell import Cell
# from h2o.fem.element.face import Face
# from h2o.fem.element.finite_element import FiniteElement
# # from h2o.pbbb.field import Field
#
# import h2o.fem.element.elements.hdgs as hdgs
# import h2o.fem.element.elements.hdg as hdg
# import h2o.fem.element.elements.stabilization as stab
#
#
# class Element:
#     cell: Cell
#     faces: List[Face]
#     gradients_operators: ndarray
#     stabilization_operator: ndarray
#     element_size: int
#     faces_indices: List[int]
#     faces_lagrange_system_row_positions: List[List[Tuple[Any, Any]]]
#     faces_lagrange_local_positions: List[List[Tuple[Any, Any]]]
#     m_cell_cell_inv: ndarray
#     m_cell_faces: ndarray
#     v_cell: ndarray
#     cell_unknown_vector: ndarray
#
#     def __init__(
#         self,
#         field: Field,
#         finite_element: FiniteElement,
#         cell: Cell,
#         faces: List[Face],
#         faces_indices: List[int],
#         faces_lagrange_system_row_positions: List[List[Tuple[Any, Any]]],
#         faces_lagrange_system_col_positions: List[List[Tuple[Any, Any]]],
#         faces_lagrange_local_positions: List[List[Tuple[Any, Any]]],
#     ):
#         """
#
#         Args:
#             field:
#             finite_element:
#             cell:
#             faces:
#             faces_indices:
#             faces_lagrange_positions:
#             faces_lag_local_positions:
#         """
#         self.cell = cell
#         self.faces = faces
#         self.faces_indices = faces_indices
#         self.faces_lagrange_system_row_positions = faces_lagrange_system_row_positions
#         self.faces_lagrange_system_col_positions = faces_lagrange_system_col_positions
#         self.faces_lagrange_local_positions = faces_lagrange_local_positions
#         _dx = field.field_dimension
#         _cl = finite_element.cell_basis_l.dimension
#         _fk = finite_element.face_basis_k.dimension
#         _nf = len(self.faces)
#         self.element_size = _dx * (_cl + _nf * _fk)
#         self.m_cell_cell_inv = np.zeros((_cl * _dx, _cl * _dx), dtype=real)
#         self.m_cell_faces = np.zeros((_cl * _dx, _nf * _fk * _dx), dtype=real)
#         self.v_cell = np.zeros((_cl * _dx,), dtype=real)
#         if finite_element.element_type in [ElementType.HDG_LOW, ElementType.HDG_EQUAL, ElementType.HDG_HIGH]:
#             if field.derivation_type == DerivationType.SYMMETRIC:
#                 self.gradients_operators = hdgs.get_gradient_operators(field, finite_element, cell, faces)
#                 # self.stabilization_operator = hdgs.get_stabilization_operator(field, finite_element, cell, faces)
#                 self.stabilization_operator = stab.get_stab_test(field, finite_element, cell, faces)
#                 # self.stabilization_operator = stab.get_stabilization_operator(field, finite_element, cell, faces)
#             elif field.derivation_type == DerivationType.FULL:
#                 self.gradients_operators = hdg.get_gradient_operators(field, finite_element, cell, faces)
#                 # self.stabilization_operator = hdg.get_stabilization_operator(field, finite_element, cell, faces)
#                 self.stabilization_operator = stab.get_stab_test(field, finite_element, cell, faces)
#                 # self.stabilization_operator = stab.get_stabilization_operator(field, finite_element, cell, faces)
#             else:
#                 raise KeyError("NO")
#         else:
#             raise KeyError("NO")
#         self.cell_unknown_vector = np.zeros((_dx * _cl,), dtype=real)
#         return
#
#     def get_element_unknown_vector(
#         self, field: Field, finite_element: FiniteElement, faces_unknown_vector: ndarray
#     ) -> ndarray:
#         """
#
#         Args:
#             field:
#             finite_element:
#             faces_unknown_vector:
#
#         Returns:
#
#         """
#         _dx = field.field_dimension
#         _fk = finite_element.face_basis_k.dimension
#         _cl = finite_element.cell_basis_l.dimension
#         element_unknown_vector = np.zeros((self.element_size,), dtype=real)
#         for _i_local, _i_global in enumerate(self.faces_indices):
#             _c0_fg = _i_global * (_fk * _dx)
#             _c1_fg = (_i_global + 1) * (_fk * _dx)
#             _c0_fl = (_dx * _cl) + _i_local * (_fk * _dx)
#             _c1_fl = (_dx * _cl) + (_i_local + 1) * (_fk * _dx)
#             element_unknown_vector[_c0_fl:_c1_fl] += faces_unknown_vector[_c0_fg:_c1_fg]
#         _c0_c = _cl * _dx
#         element_unknown_vector[:_c0_c] += self.cell_unknown_vector
#         return element_unknown_vector
#
#     def get_transformation_gradient(
#         self, field: Field, finite_element: FiniteElement, faces_unknown_vector: ndarray, _qc: int
#     ):
#         """
#
#         Args:
#             field:
#             finite_element:
#             faces_unknown_vector:
#             _qc:
#
#         Returns:
#
#         """
#         element_unknown_vector = self.get_element_unknown_vector(field, finite_element, faces_unknown_vector)
#         if field.strain_type == StrainType.DISPLACEMENT_TRANSFORMATION_GRADIENT:
#             ones_vect = np.zeros((field.gradient_dimension,), dtype=real)
#             for indx in range(3):
#                 ones_vect[indx] += 1.0
#             # transformation_gradient = ones_vect + self.gradients_operators[_qc] @ element_unknown_vector
#             transformation_gradient_0 = self.gradients_operators[_qc] @ element_unknown_vector
#             # transformation_gradient_0[1:] = np.zeros((field.gradient_dimension-1,))
#             transformation_gradient = ones_vect + transformation_gradient_0
#         elif field.strain_type == StrainType.DISPLACEMENT_SYMMETRIC_GRADIENT:
#             # transformation_gradient = self.gradients_operators[_qc] @ element_unknown_vector
#             transformation_gradient_0 = self.gradients_operators[_qc] @ element_unknown_vector
#             # transformation_gradient_0[-1] = 0.0
#             # transformation_gradient_0[1] = 0.0
#             transformation_gradient = transformation_gradient_0
#         else:
#             raise KeyError("No such strain type")
#         return transformation_gradient
#
#     def get_cell_field_value(
#         self, field: Field, finite_element: FiniteElement, faces_unknown_vector: ndarray, point: ndarray, direction: int
#     ) -> float:
#         """
#
#         Args:
#             field:
#             finite_element:
#             faces_unknown_vector:
#             point:
#             direction:
#
#         Returns:
#
#         """
#         element_unknown_vector = self.get_element_unknown_vector(field, finite_element, faces_unknown_vector)
#         _cl = finite_element.cell_basis_l.dimension
#         _c0 = direction * _cl
#         _c1 = (direction + 1) * _cl
#         v = finite_element.cell_basis_l.evaluate_function(point, self.cell.shape.centroid, self.cell.shape.diameter)
#         field_unknown_value = v @ element_unknown_vector[_c0:_c1]
#         return field_unknown_value
#
#     # def compute_transformation_gradient(self, field: Field, _qc: int, element_unknown_increment: ndarray):
#     #     """
#     #
#     #     Args:
#     #         field:
#     #         _qc:
#     #         element_unknown_increment:
#     #
#     #     Returns:
#     #
#     #     """
#     #     if field.strain_type == StrainType.DISPLACEMENT_TRANSFORMATION_GRADIENT:
#     #         ones_vect = np.zeros((field.gradient_dimension,))
#     #         for indx in range(3):
#     #             ones_vect[indx] += 1.0
#     #         transformation_gradient = ones_vect + self.gradients_operators[_qc] @ element_unknown_increment
#     #         # transformation_gradient = ones_vect + np.copy(self.gradients_operators[_qc]) @ np.copy(element_unknown_increment)
#     #     elif field.strain_type == StrainType.DISPLACEMENT_SYMMETRIC_GRADIENT:
#     #         transformation_gradient = self.gradients_operators[_qc] @ element_unknown_increment
#     #         # transformation_gradient = np.copy(self.gradients_operators[_qc]) @ np.copy(element_unknown_increment)
#     #     else:
#     #         raise KeyError("No such strain type")
#     #     return transformation_gradient
#
#     # def make_condensation(
#     #     self, field: Field, finite_element: FiniteElement, matrix: ndarray, vector: ndarray
#     # ) -> (ndarray, ndarray):
#     #     """
#     #
#     #     Args:
#     #         field:
#     #         finite_element:
#     #         matrix:
#     #         vector:
#     #
#     #     Returns:
#     #
#     #     """
#     #     _cl = finite_element.cell_basis_l.dimension
#     #     _dx = field.field_dimension
#     #     _col_stop = _dx * _cl
#     #     m_cell_cell = matrix[:_col_stop, :_col_stop]
#     #     m_cell_faces = matrix[:_col_stop, _col_stop:]
#     #     m_faces_cell = matrix[_col_stop:, :_col_stop]
#     #     m_faces_faces = matrix[_col_stop:, _col_stop:]
#     #     v_cell = vector[:_col_stop]
#     #     v_faces = vector[_col_stop:]
#     #     m_cell_cell_inv = np.linalg.inv(m_cell_cell)
#     #     ge = m_faces_cell @ m_cell_cell_inv
#     #     gd = ge @ m_cell_faces
#     #     matrix_cond = m_faces_faces - gd
#     #     vector_cond = v_faces - ge @ v_cell
#     #     self.m_cell_cell_inv = m_cell_cell_inv
#     #     self.m_cell_faces = m_cell_faces
#     #     self.v_cell = v_cell
#     #     return matrix_cond, vector_cond
#
#     # def get_element_unknwon_correction(
#     #     self, field: Field, finite_element: FiniteElement, global_vector: ndarray
#     # ) -> ndarray:
#     #     """
#     #
#     #     Args:
#     #         field:
#     #         finite_element:
#     #         global_vector:
#     #
#     #     Returns:
#     #
#     #     """
#     #     _fk = finite_element.face_basis_k.dimension
#     #     _dx = field.field_dimension
#     #     _cl = finite_element.cell_basis_l.dimension
#     #     _nf = len(self.faces)
#     #     element_correction = np.zeros((self.element_size,))
#     #     face_correction = np.zeros((_nf * _fk * _dx,))
#     #     for i, k in enumerate(self.faces_indices):
#     #         _c0_g = k * _fk * _dx
#     #         _c1_g = (k + 1) * _fk * _dx
#     #         _c0_l = i * _fk * _dx
#     #         _c1_l = (i + 1) * _fk * _dx
#     #         face_correction[_c0_l:_c1_l] += global_vector[_c0_g:_c1_g]
#     #     cell_correction = self.m_cell_cell_inv @ (self.v_cell - self.m_cell_faces @ face_correction)
#     #     _c0 = _dx * _cl
#     #     element_correction[:_c0] += cell_correction
#     #     element_correction[_c0:] += face_correction
#     #     return element_correction
#     #
#     # def get_face_unknown_increment(
#     #     self, field: Field, finite_element: FiniteElement, global_vector: ndarray
#     # ) -> ndarray:
#     #     """
#     #
#     #     Args:
#     #         field:
#     #         finite_element:
#     #         global_vector:
#     #
#     #     Returns:
#     #
#     #     """
#     #     _fk = finite_element.face_basis_k.dimension
#     #     _dx = field.field_dimension
#     #     _nf = len(self.faces)
#     #     face_increment = np.zeros((_nf * _fk * _dx,))
#     #     for i, k in enumerate(self.faces_indices):
#     #         _col_strt = k * _fk * _dx
#     #         _col_stop = (k + 1) * _fk * _dx
#     #         _col_0 = i * _fk * _dx
#     #         _col_1 = (i + 1) * _fk * _dx
#     #         face_increment[_col_0:_col_1] = global_vector[_col_strt:_col_stop]
#     #     return face_increment
#
#     # def get_external_forces(
#     #     self,
#     #     field: Field,
#     #     finite_element: FiniteElement,
#     #     time_step: float,
#     #     faces_boundaries_connectivity: Dict[str, List[int]],
#     #     boundary_conditions: List[BoundaryCondition] = None,
#     #     loads: List[Load] = None,
#     # ) -> ndarray:
#     #     """
#     #
#     #     Args:
#     #         field:
#     #         finite_element:
#     #         time_step:
#     #         faces_boundaries_connectivity:
#     #         boundary_conditions:
#     #         loads:
#     #
#     #     Returns:
#     #
#     #     """
#     #     # fe = finite_element
#     #     _dx = field.field_dimension
#     #     _cl = finite_element.cell_basis_l.dimension
#     #     _fk = finite_element.face_basis_k.dimension
#     #     _nf = len(self.faces)
#     #     _es = _dx * (_cl + _nf * _fk)
#     #     external_forces = np.zeros((_es,), dtype=real)
#     #     if loads is None:
#     #         pass
#     #     else:
#     #         for load in loads:
#     #             for qc in range(len(self.cell.quadrature_weights)):
#     #                 _x_q_c = self.cell.quadrature_points[:, qc]
#     #                 _w_q_c = self.cell.quadrature_weights[qc]
#     #                 v = finite_element.cell_basis_l.evaluate_function(
#     #                     _x_q_c, self.cell.shape.centroid, self.cell.shape.diameter
#     #                 )
#     #                 #
#     #                 vl = _w_q_c * v * load.function(time_step, _x_q_c)
#     #                 _c0 = load.direction * _cl
#     #                 _c1 = (load.direction + 1) * _cl
#     #                 external_forces[_c0:_c1] += vl
#     #     if boundary_conditions is None:
#     #         pass
#     #     else:
#     #         for boundary_condition in boundary_conditions:
#     #             if boundary_condition.boundary_type == BoundaryType.PRESSURE:
#     #                 for f_local, f_global in enumerate(self.faces_indices):
#     #                     if f_global in faces_boundaries_connectivity[boundary_condition.boundary_name]:
#     #                         for qf in range(len(self.faces[f_local].quadrature_weights)):
#     #                             _x_q_f = self.faces[f_local].quadrature_points[:, qf]
#     #                             _w_q_f = self.faces[f_local].quadrature_weights[qf]
#     #                             v = finite_element.face_basis_k.evaluate_function(
#     #                                 _x_q_f, self.faces[f_local].shape.centroid, self.faces[f_local].shape.diameter
#     #                             )
#     #                             vf = _w_q_f * v * boundary_condition.function(time_step, _x_q_f)
#     #                             _c0 = _dx * _cl + f_local * _dx * _fk + boundary_condition.direction * _fk
#     #                             _c1 = _dx * _cl + f_local * _dx * _fk + (boundary_condition.direction + 1) * _fk
#     #                             external_forces[_c0:_c1] += vf
#     #     return external_forces
#
#     # def make_condensation(self, finite_element: FiniteElement, matrix: ndarray, vector: ndarray, vector2: ndarray) -> (ndarray, ndarray):
#     #     cl = finite_element.cell_basis_l.dimension
#     #     dx = finite_element.field_dimension
#     #     col_stop = dx * cl
#     #     m_cell_cell = matrix[:col_stop, :col_stop]
#     #     m_cell_faces = matrix[:col_stop, col_stop:]
#     #     m_faces_cell = matrix[col_stop:, :col_stop]
#     #     m_faces_faces = matrix[col_stop:, col_stop:]
#     #     v_cell = vector[:col_stop]
#     #     v_faces = vector[col_stop:]
#     #     v_cell2 = vector2[:col_stop]
#     #     v_faces2 = vector2[col_stop:]
#     #     m_cell_cell_inv = np.linalg.inv(m_cell_cell)
#     #     ge = m_faces_cell @ m_cell_cell_inv
#     #     gd = ge @ m_cell_faces
#     #     matrix_cond = m_faces_faces - gd
#     #     vector_cond = v_faces - ge @ v_cell
#     #     vector2_cond = v_faces2 - ge @ v_cell2
#     #     self.m_cell_cell_inv = m_cell_cell_inv
#     #     self.m_cell_faces = m_cell_faces
#     #     self.v_cell = v_cell
#     #     # matrix_cond, vector_cond, self.m_cell_cell_inv, self.m_cell_faces, self.v_cell = condensate_system(
#     #     #     matrix, vector, finite_element
#     #     # )
#     #     return matrix_cond, vector_cond, vector2_cond
#
#     # def condensate_system(
#     #     matrix: ndarray, vector: ndarray, finite_element: FiniteElement
#     # ) -> (ndarray, ndarray, ndarray, ndarray, ndarray):
#     #     fe = finite_element
#     #     cl = fe.cell_basis_l.dimension
#     #     dx = fe.field_dimension
#     #     col_stop = dx * cl
#     #     m_cell_cell = matrix[:col_stop, :col_stop]
#     #     m_cell_faces = matrix[:col_stop, col_stop:]
#     #     m_faces_cell = matrix[col_stop:, :col_stop]
#     #     m_faces_faces = matrix[col_stop:, col_stop:]
#     #     v_cell = vector[:col_stop]
#     #     v_faces = vector[col_stop:]
#     #     m_cell_cell_inv = np.linalg.inv(m_cell_cell)
#     #     check = m_cell_cell_inv @ m_cell_cell
#     #     ge = m_faces_cell @ m_cell_cell_inv
#     #     gd = ge @ m_cell_faces
#     #     m_cond = m_faces_faces - gd
#     #     v_cond = v_faces - ge @ v_cell
#     #     return m_cond, v_cond, m_cell_cell_inv, m_cell_faces, v_cell
#
#     # def condensate_matrix(matrix: ndarray, finite_element: FiniteElement) -> ndarray:
#     #     fe = finite_element
#     #     cl = fe.cell_basis_l.dimension
#     #     dx = fe.field_dimension
#     #     col_stop = dx * cl
#     #     m_cell_cell = matrix[:col_stop, :col_stop]
#     #     m_cell_faces = matrix[:col_stop, col_stop:]
#     #     m_faces_cell = matrix[col_stop:, :col_stop]
#     #     m_faces_faces = matrix[col_stop:, col_stop:]
#     #     m_cell_cell_inv = np.linalg.inv(m_cell_cell)
#     #     ge = m_faces_cell @ m_cell_cell_inv
#     #     gd = ge @ m_cell_faces
#     #     m_cond = m_faces_faces - gd
#     #     return m_cond
#
#     # def assemble_element2(self, finite_element: FiniteElement, Fe_cond: ndarray, K_cond: ndarray, system: GlobalSystem):
#     #     fe = finite_element
#     #     fk = fe.face_basis_k.dimension
#     #     dx = fe.field_dimension
#     #     for i_local, i_global in enumerate(self.faces_indices):
#     #         rg0 = (dx * fk) * i_global
#     #         rg1 = (dx * fk) * (i_global + 1)
#     #         rl0 = i_local * (dx * fk)
#     #         rl1 = (i_local + 1) * (dx * fk)
#     #         system.external_forces[rg0:rg1] += Fe_cond[rl0:rl1]
#     #         for j_local, j_global in enumerate(self.faces_indices):
#     #             cg0 = j_global * (dx * fk)
#     #             cg1 = (j_global + 1) * (dx * fk)
#     #             cl0 = j_local * (dx * fk)
#     #             cl1 = (j_local + 1) * (dx * fk)
#     #             system.global_matrix[rg0:rg1, cg0:cg1] += K_cond[rl0:rl1, cl0:cl1]
#     #     return
#
#     # def assemble_element(self, finite_element: FiniteElement, vector: ndarray, matrix: ndarray, system: GlobalSystem):
#     #     # K_cond, B_cond, self.m_cell_cell_inv, self.m_cell_faces, self.v_cell = condensate_system(
#     #     #     matrix, vector, finite_element
#     #     # )
#     #     K_cond, B_cond = self.make_condensation(finite_element, vector, matrix)
#     #     fe = finite_element
#     #     fk = fe.face_basis_k.dimension
#     #     dx = fe.field_dimension
#     #     for i_local, i_global in enumerate(self.faces_indices):
#     #         rg0 = (dx * fk) * i_global
#     #         rg1 = (dx * fk) * (i_global + 1)
#     #         rl0 = i_local * (dx * fk)
#     #         rl1 = (i_local + 1) * (dx * fk)
#     #         system.global_vector[rg0:rg1] += B_cond[rl0:rl1]
#     #         for j_local, j_global in enumerate(self.faces_indices):
#     #             cg0 = j_global * (dx * fk)
#     #             cg1 = (j_global + 1) * (dx * fk)
#     #             cl0 = j_local * (dx * fk)
#     #             cl1 = (j_local + 1) * (dx * fk)
#     #             system.global_matrix[rg0:rg1, cg0:cg1] += K_cond[rl0:rl1, cl0:cl1]
#     #     return
#
#     # def get_internal_forces(self, element_increment: ndarray) -> ndarray:
#     #     # nq = len(self.cell.quadrature_weights)
#     #     # fe = finite_element
#     #     # strains = np.array((nq, fe.gradient_dimension,))
#     #     internal_forces = np.zeros(self.element_size,)
#     #     for qc in range(len(self.cell.quadrature_weights)):
#     #         # strains[qc] = self.gradients_operators[qc] @ element_increment
#     #         # self.material_points.strains[qc] = self.gradients_operators[qc] @ element_increment
#     #         # self.material_points.stresses[qc] = (
#     #         #     # self.material_points.tangent_operators[qc] @ self.material_points.strains[qc]
#     #         #     self.material_points.tangent_operators[qc]
#     #         #     @ self.gradients_operators[qc]
#     #         #     @ element_increment
#     #         # )
#     #         stress_at_qc = self.material_points.tangent_operators[qc] @ (
#     #             self.gradients_operators[qc] @ element_increment
#     #         )
#     #         # internal_forces += self.gradients_operators[qc].T @ self.material_points.stresses[qc]
#     #         internal_forces += self.gradients_operators[qc].T @ stress_at_qc
#     #     internal_forces += self.stabilization_operator @ element_increment
#     #     return internal_forces
#     #
#     # def get_internal_forces2(self, finite_element: FiniteElement, global_vector: ndarray) -> ndarray:
#     #     # nq = len(self.cell.quadrature_weights)
#     #     # fe = finite_element
#     #     # strains = np.array((nq, fe.gradient_dimension,))
#     #     element_increment = self.get_element_increment(finite_element, global_vector)
#     #     internal_forces = np.zeros(self.element_size,)
#     #     for qc in range(len(self.cell.quadrature_weights)):
#     #         # strains[qc] = self.gradients_operators[qc] @ element_increment
#     #         # self.material_points.strains[qc] = self.gradients_operators[qc] @ element_increment
#     #         # self.material_points.stresses[qc] = (
#     #         #     # self.material_points.tangent_operators[qc] @ self.material_points.strains[qc]
#     #         #     self.material_points.tangent_operators[qc]
#     #         #     @ self.gradients_operators[qc]
#     #         #     @ element_increment
#     #         # )
#     #         stress_at_qc = self.material_points.tangent_operators[qc] @ (
#     #             self.gradients_operators[qc] @ element_increment
#     #         )
#     #         # internal_forces += self.gradients_operators[qc].T @ self.material_points.stresses[qc]
#     #         internal_forces += self.gradients_operators[qc].T @ stress_at_qc
#     #     internal_forces += 1.0 * self.stabilization_operator @ element_increment
#     #     return internal_forces
#     #
#     # def get_internal_forces3(self, finite_element: FiniteElement, global_vector: ndarray) -> ndarray:
#     #     fe = finite_element
#     #     fk = fe.face_basis_k.dimension
#     #     dx = fe.field_dimension
#     #     nf = len(self.faces)
#     #     face_size = nf * fk * dx
#     #     face_increment = self.get_face_increment(global_vector)
#     #     # internal_forces = np.zeros((face_size,))
#     #     K_loc = np.zeros((face_size, face_size))
#     #     for qc in range(len(self.cell.quadrature_weights)):
#     #         K_loc += (
#     #             self.gradients_operators[qc].T
#     #             @ self.material_points.tangent_operators[qc]
#     #             @ (self.gradients_operators[qc])
#     #         )
#     #     K_loc += 1.0 * self.stabilization_operator
#     #     K_loc_cond = condensate_matrix(K_loc, finite_element)
#     #     internal_forces = K_loc_cond @ face_increment
#     #     return internal_forces
#
#     # def get_internal_forces2(self, finite_element: FiniteElement, global_vector: ndarray) -> ndarray:
#     #     # nq = len(self.cell.quadrature_weights)
#     #     # fe = finite_element
#     #     # strains = np.array((nq, fe.gradient_dimension,))
#     #     element_increment = self.get_element_increment(finite_element, global_vector)
#     #     internal_forces = np.zeros(self.element_size,)
#     #     for qc in range(len(self.cell.quadrature_weights)):
#     #         # strains[qc] = self.gradients_operators[qc] @ element_increment
#     #         # self.material_points.strains[qc] = self.gradients_operators[qc] @ element_increment
#     #         # self.material_points.stresses[qc] = (
#     #         #     # self.material_points.tangent_operators[qc] @ self.material_points.strains[qc]
#     #         #     self.material_points.tangent_operators[qc]
#     #         #     @ self.gradients_operators[qc]
#     #         #     @ element_increment
#     #         # )
#     #         stress_at_qc = self.material_points.tangent_operators[qc] @ self.gradients_operators[qc] @ element_increment
#     #         # internal_forces += self.gradients_operators[qc].T @ self.material_points.stresses[qc]
#     #         internal_forces += self.gradients_operators[qc].T @ stress_at_qc
#     #     internal_forces += self.stabilization_operator @ element_increment
#     #     return internal_forces
#
#     # def get_residual(
#     #     self,
#     #     finite_element: FiniteElement,
#     #     # element_increment: ndarray,
#     #     global_vector: ndarray,
#     #     time_step: float,
#     #     faces_boundaries_connectivity: Dict[str, List[int]],
#     #     boundary_conditions: List[BoundaryCondition] = None,
#     #     loads: List[Load] = None,
#     # ) -> ndarray:
#     #     external_forces = self.get_external_forces(
#     #         finite_element,
#     #         time_step,
#     #         faces_boundaries_connectivity,
#     #         boundary_conditions=boundary_conditions,
#     #         loads=loads,
#     #     )
#     #     element_increment = self.get_element_increment(finite_element, global_vector)
#     #     internal_forces = self.get_internal_forces(element_increment)
#     #     # coef = 1.0
#     #     # if max(external_forces) > 0.0:
#     #     #     coef = max(external_forces)
#     #     residual = internal_forces - external_forces
#     #     return residual
#
#     # def apply_displacement_conditions(
#     #     self,
#     #     finite_element: FiniteElement,
#     #     system: GlobalSystem,
#     #     time_step: float,
#     #     faces_boundaries_connectivity: Dict[str, List[int]],
#     #     boundary_conditions: List[BoundaryCondition] = None,
#     # ):
#     #     fe = finite_element
#     #     dx = fe.field_dimension
#     #     fk = fe.face_basis_k.dimension
#     #     iter_const = 0
#     #     for boundary_condition in boundary_conditions:
#     #         if boundary_condition.boundary_type == BoundaryType.DISPLACEMENT:
#     #             for f_elemnt, f_global in enumerate(self.faces_indices):
#     #                 if f_global in faces_boundaries_connectivity[boundary_condition.boundary_name]:
#     #                     m_psi_psi_face = np.zeros((fk, fk))
#     #                     vf = np.zeros((fk,))
#     #                     for qf in range(len(self.faces[f_elemnt].quadrature_weights)):
#     #                         x_q_f = self.faces[f_elemnt].quadrature_points[:, qf]
#     #                         w_q_f = self.faces[f_elemnt].quadrature_weights[qf]
#     #                         v = fe.face_basis_k.evaluate_function(
#     #                             x_q_f, self.faces[f_elemnt].shape.centroid, self.faces[f_elemnt].shape.diameter
#     #                         )
#     #                         vf += w_q_f * v * boundary_condition.function(time_step, x_q_f)
#     #                         m_psi_psi_face += blocks.get_face_mass_matrix_in_face(
#     #                             self.faces[f_elemnt], fe.face_basis_k, fe.face_basis_k, x_q_f, w_q_f
#     #                         )
#     #                     m_psi_psi_face_inv = np.linalg.inv(m_psi_psi_face)
#     #                     vo = m_psi_psi_face_inv @ vf
#     #                     l0 = system.system_size + iter_const * fk
#     #                     l1 = system.system_size + (iter_const + 1) * fk
#     #                     c0 = f_global * dx * fk + boundary_condition.direction * fk
#     #                     c1 = (f_global * dx * fk) + (boundary_condition.direction + 1) * fk
#     #                     system.tangent_matrix[l0:l1, c0:c1] += np.eye(fk)
#     #                     system.tangent_matrix[c0:c1, l0:l1] += np.eye(fk)
#     #                     system.residual[l0:l1] += vo
#     #                     iter_const += 1
#     #     return
#
#     # def get_face_increment(self, finite_element: FiniteElement, global_vector: ndarray) -> ndarray:
#     #     fe = finite_element
#     #     dx = fe.field_dimension
#     #     fk = fe.face_basis_k.dimension
#     #     nf = len(self.faces)
#     #     face_increment = np.zeros((nf * fk * dx,))
#     #     for k in range(nf):
#     #         col_strt = k * fk * dx
#     #         col_stop = (k + 1) * fk * dx
#     #         face_increment = global_vector[col_strt:col_stop]
#     #     return face_increment
#     #
#     # def get_cell_increment(self, face_increment: ndarray) -> ndarray:
#     #     cell_increment = self.m_cell_cell_inv @ self.v_cell - self.m_cell_cell_inv @ self.m_cell_faces @ face_increment
#     #     return cell_increment
#     #
#     # def get_element_increment(self, finite_element: FiniteElement, global_vector: ndarray) -> ndarray:
#     #     fe = finite_element
#     #     dx = fe.field_dimension
#     #     cl = fe.cell_basis_l.dimension
#     #     element_increment = np.zeros((self.element_size,))
#     #     face_increment = self.get_face_increment(finite_element, global_vector)
#     #     cell_increment = self.get_cell_increment(face_increment)
#     #     col = dx * cl
#     #     element_increment[:col] += cell_increment
#     #     element_increment[col:] += face_increment
#     #     return element_increment
#
#     # def set_strains(self, element_increment: ndarray):
#     #     for qc in range(len(self.cell.quadrature_weights)):
#     #         self.material_points.strains[qc] = self.gradients_operators[qc] @ element_increment
#     #     return
#
#     # def set_stresses(self, strain: ndarray):
#     # def set_stresses(self, element_increment: ndarray):
#     #     nq = len(self.cell.quadrature_weights)
#     #     strains = np.array((nq, se))
#     #     for qc in range(len(self.cell.quadrature_weights)):
#     #         strains[qc] = self.gradients_operators[qc] @ element_increment
#     #         self.material_points.strains[qc] = self.gradients_operators[qc] @ element_increment
#     #         self.material_points.stresses[qc] = (
#     #             self.material_points.tangent_operators[qc] @ self.material_points.strains[qc]
#     #         )
#     #     return
#
#     # def get_external_forces(
#     #     self,
#     #     finite_element: FiniteElement,
#     #     time_step: float,
#     #     # loads: List[Callable] = None,
#     #     # pressures: List[Callable] = None,
#     #     faces_boundaries_connectivity: Dict[str, List[int]],
#     #     boundary_conditions: List[BoundaryCondition] = None,
#     #     loads: List[Load] = None,
#     # ):
#     #     fe = finite_element
#     #     d = fe.euclidean_dimension
#     #     dx = fe.field_dimension
#     #     cl = fe.cell_basis_l.dimension
#     #     fk = fe.face_basis_k.dimension
#     #     nf = len(self.faces)
#     #     element_size = dx * (cl + nf * fk)
#     #     external_forces = np.zeros(element_size,)
#     #     # loc_pressures = [lambda t, x: 0.0 for direction in range(dx)]
#     #     # if not bc is None:
#     #     #     for bc_item in bc:
#     #     #         if bc_item.boundary_type == BoundaryType.PRESSURE:
#     #     #             loc_pressures[bc_item.direction] = bc_item.function
#     #     #     # pressure = lambda t, x: np.zeros((dx,))
#     #     # elif isinstance(bc, list):
#     #     #     for bc_item in bc:
#     #     #         pass
#     #     loc_loads = [lambda t, x: 0.0 for direction in range(dx)]
#     #     # if not loads is None:
#     #     #     for load in loads:
#     #     #         loc_loads[load.direction] = load.function
#     #     # if isinstance(loads, list):
#     #     #     if len(loads) == dx:
#     #     #         for load in loads:
#     #     #             if isinstance(load, Load):
#     #     #                 pass
#     #     #             else:
#     #     #                 raise TypeError("load must be a Load")
#     #     #     else:
#     #     #         raise ValueError("loads must be a {}-sized list".format(dx))
#     #     # else:
#     #     #     raise TypeError("loads must be a lost of Load")
#     #     # for load in loads:
#     #     #     loc_loads[load.direction] = load.function
#     #     loc_pressures = [lambda t, x: 0.0 for direction in range(dx)]
#     #     # if loads is None:
#     #     #     # load = lambda t, x: np.zeros((dx,))
#     #     #     loads = [lambda t, x: 0.0 for direction in range(dx)]
#     #     # elif isinstance(loads, list):
#     #     #     if len(loads) == dx:
#     #     #         for load_count, load in enumerate(loads):
#     #     #             if load is None:
#     #     #                 loads[load_count] = lambda t, x: 0.0
#     #     #     else:
#     #     #         raise ValueError("loads must be a {}-sized list".format(dx))
#     #     # else:
#     #     #     raise TypeError("loads must be a list of loads")
#     #     if loads is None:
#     #         pass
#     #     else:
#     #         for load in loads:
#     #             # for i in range(dx):
#     #             for qc in range(len(self.cell.quadrature_weights)):
#     #                 x_q_c = self.cell.quadrature_points[:, qc]
#     #                 w_q_c = self.cell.quadrature_weights[qc]
#     #                 v = fe.cell_basis_l.evaluate_function(x_q_c, self.cell.shape.centroid, self.cell.shape.diameter)
#     #                 #
#     #                 vl = w_q_c * v * load.function(time_step, x_q_c)
#     #                 # col_strt = i * cl
#     #                 # col_stop = (i + 1) * cl
#     #                 col_strt = load.direction * cl
#     #                 col_stop = (load.direction + 1) * cl
#     #                 external_forces[col_strt:col_stop] += vl
#     #     if boundary_conditions is None:
#     #         pass
#     #     else:
#     #         for boundary_condition in boundary_conditions:
#     #             for k, face_index in enumerate(self.faces_indices):
#     #                 if face_index in faces_boundaries_connectivity[boundary_condition.boundary_name]:
#     #                     for qf in range(len(self.faces[k].quadrature_weights)):
#     #                         x_q_f = self.faces[k].quadrature_points[:, qf]
#     #                         w_q_f = self.faces[k].quadrature_weights[qf]
#     #                         v = fe.face_basis_k.evaluate_function(
#     #                             x_q_f, self.faces[k].shape.centroid, self.faces[k].shape.diameter
#     #                         )
#     #                         # vf = w_q_f * v * loc_pressures[i](time_step, x_q_f)
#     #                         vf = w_q_f * v * boundary_condition.function(time_step, x_q_f)
#     #                         col_strt = dx * cl + k * dx * fk + boundary_condition.direction * fk
#     #                         col_stop = dx * cl + k * dx * fk + (boundary_condition.direction + 1) * fk
#     #                         external_forces[col_strt:col_stop] += vf
#     #         # for k, face in enumerate(self.faces):
#     #         #     for qf in range(len(face.quadrature_weights)):
#     #         #         x_q_f = face.quadrature_points[:, qf]
#     #         #         w_q_f = face.quadrature_weights[qf]
#     #         #         v = fe.face_basis_k.evaluate_function(x_q_f, face.shape.centroid, face.shape.diameter)
#     #         #         vf = w_q_f * v * loc_pressures[i](time_step, x_q_f)
#     #         #         col_strt = dx * cl + k * dx * fk + i * fk
#     #         #         col_stop = dx * cl + k * dx * fk + (i + 1) * fk
#     #         #         external_forces[col_strt:col_stop] += vf
#     #     return external_forces
