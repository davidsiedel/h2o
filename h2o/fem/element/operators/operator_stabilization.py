from h2o.h2o import *
from h2o.geometry.shape import Shape, get_rotation_matrix
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement

def get_stabilization_operator2(field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape]) -> ndarray:
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    _io = finite_element.construction_integration_order
    x_c = cell.centroid
    h_c = cell.diameter
    stabilization_operator = np.zeros((_es, _es), dtype=real)
    stabilization_op = np.zeros((_fk * _dx, _es), dtype=real)
    for _f, face in enumerate(faces):
        h_f = face.diameter
        x_f = face.centroid
        face_quadrature_points = face.get_quadrature_points(_io)
        face_quadrature_weights = face.get_quadrature_weights(_io)
        face_quadrature_size = face.get_quadrature_size(_io)
        face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
        m_mas_f = np.zeros((_fk, _fk), dtype=real)
        m_hyb_f = np.zeros((_fk, _cl), dtype=real)
        for qf in range(face_quadrature_size):
            x_q_f = face_quadrature_points[:, qf]
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            x_q_f_p = face_rotation_matrix @ x_q_f
            x_c_p = face_rotation_matrix @ x_c
            phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, h_c)
            # phi_l = cell_basis_l.get_phi_vector(x_q_f_p, x_c_p, h_c)
            psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, h_f)
            m_hyb_f += w_q_f * np.tensordot(psi_k, phi_l, axes=0)
            m_mas_f += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
        m_mas_f_inv = np.linalg.inv(m_mas_f)
        proj_mat = m_mas_f_inv @ m_hyb_f
        m = np.eye(_fk, dtype=real)
        m_mas_f2 = np.zeros((_dx * _fk, _dx * _fk), dtype=real)
        for _i in range(_dx):
            _ri = _i * _fk
            _rj = (_i + 1) * _fk
            _ci = _i * _cl
            _cj = (_i + 1) * _cl
            stabilization_op[_ri:_rj, _ci:_cj] -= proj_mat
            _ci = _dx * _cl + _f * _dx * _fk + _i * _fk
            _cj = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
            stabilization_op[_ri:_rj, _ci:_cj] += m
            m_mas_f2[_ri:_rj,_ri:_rj] += m_mas_f
        stabilization_operator += (1.0 / h_f) * stabilization_op.T @ m_mas_f2 @ stabilization_op
        # stabilization_operator += stabilization_op.T @ m_mas_f @ stabilization_op
    return stabilization_operator


def get_stabilization_operator_component(
    field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], _f: int, _i: int
) -> ndarray:
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    _io = finite_element.construction_integration_order
    x_c = cell.centroid
    h_c = cell.diameter
    stabilization_operator = np.zeros((_es, _es), dtype=real)
    stabilization_op = np.zeros((_fk, _es), dtype=real)
    face = faces[_f]
    h_f = face.diameter
    x_f = face.centroid
    face_quadrature_points = face.get_quadrature_points(_io)
    face_quadrature_weights = face.get_quadrature_weights(_io)
    face_quadrature_size = face.get_quadrature_size(_io)
    face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
    m_mas_f = np.zeros((_fk, _fk), dtype=real)
    m_hyb_f = np.zeros((_fk, _cl), dtype=real)
    for qf in range(face_quadrature_size):
        x_q_f = face_quadrature_points[:, qf]
        w_q_f = face_quadrature_weights[qf]
        s_f = (face_rotation_matrix @ x_f)[:-1]
        s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
        x_q_f_p = face_rotation_matrix @ x_q_f
        x_c_p = face_rotation_matrix @ x_c
        phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, h_c)
        # phi_l = cell_basis_l.get_phi_vector(x_q_f_p, x_c_p, h_c)
        psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, h_f)
        m_hyb_f += w_q_f * np.tensordot(psi_k, phi_l, axes=0)
        m_mas_f += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
    m_mas_f_inv = np.linalg.inv(m_mas_f)
    proj_mat = m_mas_f_inv @ m_hyb_f
    m = np.eye(_fk, dtype=real)
    _ci = _i * _cl
    _cj = (_i + 1) * _cl
    stabilization_op[:, _ci:_cj] -= proj_mat
    _ci = _dx * _cl + _f * _dx * _fk + _i * _fk
    _cj = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
    stabilization_op[:, _ci:_cj] += m
    # stabilization_operator += (1.0 / h_f) * stabilization_op.T @ m_mas_f @ stabilization_op
    stabilization_operator += stabilization_op.T @ m_mas_f @ stabilization_op
    return stabilization_operator


def get_stabilization_operator(field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape]) -> ndarray:
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    stabilization_operator = np.zeros((_es, _es), dtype=real)
    for _f, face in enumerate(faces):
        h_f = face.diameter
        for _i in range(_dx):
            stabilization_operator += get_stabilization_operator_component(field, finite_element, cell, faces, _f, _i)
        stabilization_operator *= 1.0 / h_f
    return stabilization_operator
