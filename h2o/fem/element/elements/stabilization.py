from typing import List

from h2o.fem.element.elements.blocks import *
from h2o.fem.element.finite_element import FiniteElement
from h2o.pbbb.field import Field


def get_stabilization_operator(field: Field, finite_element: FiniteElement, cell: Cell, faces: List[Face]):
    """

    Args:
        field:
        finite_element:
        cell:
        faces:

    Returns:

    """
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    stabilization_operator = np.zeros((_es, _es), dtype=real)
    for i in range(_dx):
        for k, face in enumerate(faces):
            stabilization_op = np.zeros((_fk, _es), dtype=real)
            m_mas_f = np.zeros((_fk, _fk), dtype=real)
            m_hyb_f = np.zeros((_fk, _cl), dtype=real)
            for qf in range(len(face.quadrature_weights)):
                x_q_f = face.quadrature_points[:, qf]
                w_q_f = face.quadrature_weights[qf]
                m_hyb_f_T = get_hybrid_mass_matrix_in_face(
                    cell, face, finite_element.cell_basis_l, finite_element.face_basis_k, x_q_f, w_q_f
                )
                m_hyb_f += m_hyb_f_T.T
                m_mas_f += get_face_mass_matrix_in_face(
                    face, finite_element.face_basis_k, finite_element.face_basis_k, x_q_f, w_q_f
                )
            col_strt = i * _cl
            col_stop = (i + 1) * _cl
            m_mas_f_inv = np.linalg.inv(m_mas_f)
            if debug_mode == 0:
                print("FACE MASS MATRIX IN STABILIZATION COND :")
                print("{}".format(np.linalg.cond(m_mas_f)))
            stabilization_op[:, col_strt:col_stop] -= m_mas_f_inv @ m_hyb_f
            # stabilization_op[:, col_strt:col_stop] -= m_mas_f @ m_hyb_f
            col_strt = _dx * _cl + k * _dx * _fk + i * _fk
            col_stop = _dx * _cl + k * _dx * _fk + (i + 1) * _fk
            m = np.eye(_fk, dtype=real)
            stabilization_op[:, col_strt:col_stop] += m
            stabilization_operator += (1.0 / face.shape.diameter) * stabilization_op.T @ m_mas_f @ stabilization_op
            # stabilization_operator += stabilization_op.T @ m_mas_f @ stabilization_op
            # if i == 0:
            #     stabilization_operator *= (1.0 / face.shape.diameter)
            # stabilization_operator += (1.0 / face.shape.diameter) * stabilization_op.T @ m_mas_f_inv @ stabilization_op
    # for k, face in enumerate(faces):
    #     stabilization_operator = (1.0 / face.shape.diameter) * stabilization_operator
    return stabilization_operator

def get_stab_test(field: Field, finite_element: FiniteElement, cell: Cell, faces: List[Face])-> ndarray:
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    stabilization_operator = np.zeros((_es, _es), dtype=real)
    for _f, face in enumerate(faces):
        face_stabilization_operator = np.zeros((_es, _es), dtype=real)
        stabilization_vector_operator = np.zeros((_dx, _es), dtype=real)
        x_f = face.shape.centroid
        h_f = face.shape.diameter
        m_mas_f = np.zeros((_fk, _fk), dtype=real)
        m_hyb_f = np.zeros((_fk, _cl), dtype=real)
        # m_prj_f = np.zeros((_fk, _cl))
        for qf in range(len(face.quadrature_weights)):
            x_q_f = face.quadrature_points[:, qf]
            w_q_f = face.quadrature_weights[qf]
            m_mas_f += get_face_mass_matrix_in_face(
                face, finite_element.face_basis_k, finite_element.face_basis_k, x_q_f, w_q_f
            )
            m_hyb_f += get_test_mass_matrix_in_face(
                cell, face, finite_element.cell_basis_l, finite_element.face_basis_k, x_q_f, w_q_f
            )
        m_mas_f_inv = np.linalg.inv(m_mas_f)
        if debug_mode == 0:
            print("FACE MASS MATRIX IN STABILIZATION COND :")
            print("{}".format(np.linalg.cond(m_mas_f)))
        m_prj_f = m_mas_f_inv @ m_hyb_f
        m_eye_f = np.eye(_fk, dtype=real)
        for _x in range(_dx):
            stabilization_vector_component_op = np.zeros((_fk, _es), dtype=real)
            c0 = _x * _cl
            c1 = (_x + 1) * _cl
            stabilization_vector_component_op[:,c0:c1] -= m_prj_f
            c0 = _dx * _cl + _f * _dx * _fk + _x * _fk
            c1 = _dx * _cl + _f * _dx * _fk + (_x + 1) * _fk
            stabilization_vector_component_op[:,c0:c1] += m_eye_f
            for qf in range(len(face.quadrature_weights)):
                x_q_f = face.quadrature_points[:, qf]
                w_q_f = face.quadrature_weights[qf]
                s_q_f = (face.mapping_matrix @ x_q_f)[:-1]
                s_f = (face.mapping_matrix @ x_f)[:-1]
                v_face = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, h_f)
                # stabilization_vector_component_at_quad = v_face @ stabilization_vector_component_op
                stabilization_vector_operator[_x,:] += v_face @ stabilization_vector_component_op
        for qf in range(len(face.quadrature_weights)):
            x_q_f = face.quadrature_points[:, qf]
            w_q_f = face.quadrature_weights[qf]
            m_eye_tan = np.eye(_dx, dtype=real)
            face_stabilization_operator += w_q_f * stabilization_vector_operator.T @ m_eye_tan @ stabilization_vector_operator
        weighted_face_stabilization_operator = (1.0/h_f) * face_stabilization_operator
        stabilization_operator += weighted_face_stabilization_operator
    return stabilization_operator


# for i in range(dx):
#     for j in range(d):
#         # if i != j:
#         m_mas = np.zeros((ck, ck), dtype=real)
#         for qc in range(len(cell.quadrature_weights)):
#             x_q_c = cell.quadrature_points[:, qc]
#             w_q_c = cell.quadrature_weights[qc]
#             m_mas += get_cell_mass_matrix_in_cell(cell, fe.cell_basis_k, fe.cell_basis_k, x_q_c, w_q_c)
#         m_mas_inv = np.linalg.inv(m_mas)
#         local_grad_matric = np.zeros((ck, element_size), dtype=real)
#         m_adv_j = np.zeros((ck, cl), dtype=real)
#         m_adv_i = np.zeros((ck, cl), dtype=real)
#         for qc in range(len(cell.quadrature_weights)):
#             x_q_c = cell.quadrature_points[:, qc]
#             w_q_c = cell.quadrature_weights[qc]
#             m_adv_j += get_cell_advection_matrix_in_cell(cell, fe.cell_basis_k, fe.cell_basis_l, j, x_q_c, w_q_c)
#             m_adv_i += get_cell_advection_matrix_in_cell(cell, fe.cell_basis_k, fe.cell_basis_l, i, x_q_c, w_q_c)
#         _c0 = i * cl
#         _c1 = (i + 1) * cl
#         local_grad_matric[:, _c0:_c1] += m_adv_j
#         _c0 = j * cl
#         _c1 = (j + 1) * cl
#         local_grad_matric[:, _c0:_c1] += m_adv_i
#         for k, face in enumerate(faces):
#             dist_in_face = (face.mapping_matrix @ (face.shape.centroid - cell.shape.centroid))[-1]
#             # sign = dist[-1]
#             if dist_in_face > 0:
#                 # normal_vector = face.mapping_matrix[:, -1]
#                 normal_vector_component_j = face.mapping_matrix[-1, j]
#                 normal_vector_component_i = face.mapping_matrix[-1, i]
#             else:
#                 # normal_vector = -face.mapping_matrix[:, -1]
#                 normal_vector_component_j = -face.mapping_matrix[-1, j]
#                 normal_vector_component_i = -face.mapping_matrix[-1, i]
#             m_mas_f = np.zeros((ck, cl), dtype=real)
#             m_hyb_f = np.zeros((ck, fk), dtype=real)
#             for qf in range(len(face.quadrature_weights)):
#                 x_q_f = face.quadrature_points[:, qf]
#                 w_q_f = face.quadrature_weights[qf]
#                 m_mas_f += get_cell_mass_matrix_in_face(cell, fe.cell_basis_k, fe.cell_basis_l, x_q_f, w_q_f)
#                 m_hyb_f += get_hybrid_mass_matrix_in_face(
#                     cell, face, fe.cell_basis_k, fe.face_basis_k, x_q_f, w_q_f
#                 )
#             _c0 = i * cl
#             _c1 = (i + 1) * cl
#             local_grad_matric[:, _c0:_c1] -= m_mas_f * normal_vector_component_j
#             _c0 = dx * cl + k * dx * fk + i * fk
#             _c1 = dx * cl + k * dx * fk + (i + 1) * fk
#             local_grad_matric[:, _c0:_c1] += m_hyb_f * normal_vector_component_j
#             _c0 = j * cl
#             _c1 = (j + 1) * cl
#             local_grad_matric[:, _c0:_c1] -= m_mas_f * normal_vector_component_i
#             _c0 = dx * cl + k * dx * fk + j * fk
#             _c1 = dx * cl + k * dx * fk + (j + 1) * fk
#             local_grad_matric[:, _c0:_c1] += m_hyb_f * normal_vector_component_i
#         local_grad_matric = (1.0 / 2.0) * m_mas_inv @ local_grad_matric
#         line = fe.voigt_indices[(i, j)]
#         for qc in range(len(cell.quadrature_weights)):
#             x_q_c = cell.quadrature_points[:, qc]
#             v = fe.cell_basis_k.evaluate_function(x_q_c, cell.shape.centroid, cell.shape.diameter)
#             gradient_component = fe.voigt_coefficients[(i,j)] * v @ local_grad_matric
#             gradient_operators[qc, line] += gradient_component
# gradient_operators = np.zeros(gradient_operators.shape)
# return gradient_operators

# def get_stabilization_operator(element_data: FiniteElement, cell: Cell, faces: List[Face]):
#     ef = element_data
#     dx = ef.field_dimension
#     cl = ef.cell_basis_l.dimension
#     fk = ef.face_basis_k.dimension
#     nf = len(faces)
#     element_size = dx * (cl + nf * fk)
#     stabilization_operator = np.zeros((element_size, element_size))
#     for i in range(dx):
#         for k, face in enumerate(faces):
#             stabilization_op = np.zeros((fk, element_size))
#             m_mas_f = np.zeros((fk, fk))
#             m_hyb_f = np.zeros((fk, cl))
#             for qf in range(len(face.quadrature_weights)):
#                 x_q_f = face.quadrature_points[:, qf]
#                 w_q_f = face.quadrature_weights[qf]
#                 m_hyb_f_T = get_hybrid_mass_matrix_in_face(cell, face, ef.cell_basis_l, ef.face_basis_k, x_q_f, w_q_f)
#                 m_hyb_f += m_hyb_f_T.T
#                 m_mas_f += get_face_mass_matrix_in_face(face, ef.face_basis_k, ef.face_basis_k, x_q_f, w_q_f)
#             col_strt = i * cl
#             col_stop = (i + 1) * cl
#             m_mas_f_inv = np.linalg.inv(m_mas_f)
#             stabilization_op[:, col_strt:col_stop] -= m_mas_f_inv @ m_hyb_f
#             col_strt = dx * cl + k * dx * fk + i * fk
#             col_stop = dx * cl + k * dx * fk + (i + 1) * fk
#             m = np.eye(fk)
#             stabilization_op[:, col_strt:col_stop] += m
#             # stabilization_operator += (1.0 / face.shape.diameter) * stabilization_op.T @ m_mas_f @ stabilization_op
#             stabilization_operator += (1.0 / face.shape.diameter) * stabilization_op.T @ m_mas_f_inv @ stabilization_op
#     return stabilization_operator
