from h2o.h2o import *
from h2o.fem.element.finite_element import FiniteElement
from h2o.geometry.shape import Shape, get_rotation_matrix
from h2o.field.field import Field


def get_symmetric_gradient_component_matrix(
    field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], _i: int, _j: int
) -> ndarray:
    _d = field.euclidean_dimension
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _ck = finite_element.cell_basis_k.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    _io = finite_element.construction_integration_order
    # --- CELL ENVIRONMENT
    _c_is = cell.get_quadrature_size(_io)
    cell_quadrature_points = cell.get_quadrature_points(_io)
    cell_quadrature_weights = cell.get_quadrature_weights(_io)
    x_c = cell.get_centroid()
    h_c = cell.get_diameter()
    local_grad_matric = np.zeros((_ck, _es), dtype=real)
    m_mas = np.zeros((_ck, _ck), dtype=real)
    m_adv_j = np.zeros((_ck, _cl), dtype=real)
    m_adv_i = np.zeros((_ck, _cl), dtype=real)
    for qc in range(_c_is):
        x_q_c = cell_quadrature_points[:, qc]
        w_q_c = cell_quadrature_weights[qc]
        phi_k = finite_element.cell_basis_k.evaluate_function(x_q_c, x_c, h_c)
        d_phi_l_j = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, h_c, _j)
        d_phi_l_i = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, h_c, _i)
        m_adv_j += w_q_c * np.tensordot(phi_k, d_phi_l_j, axes=0)
        m_adv_i += w_q_c * np.tensordot(phi_k, d_phi_l_i, axes=0)
        m_mas += w_q_c * np.tensordot(phi_k, phi_k, axes=0)
    m_mas_inv = np.linalg.inv(m_mas)
    _c0 = _i * _cl
    _c1 = (_i + 1) * _cl
    local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * m_adv_j
    _c0 = _j * _cl
    _c1 = (_j + 1) * _cl
    local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * m_adv_i
    for _f, face in enumerate(faces):
        # --- FACE ENVIRONMENT
        _f_is = face.get_quadrature_size(_io)
        face_quadrature_points = face.get_quadrature_points(_io)
        face_quadrature_weights = face.get_quadrature_weights(_io)
        x_f = face.get_centroid()
        h_f = face.get_diameter()
        face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
        dist_in_face = (face_rotation_matrix @ (x_f - x_c))[-1]
        if dist_in_face > 0:
            normal_vector_component_j = face_rotation_matrix[-1, _j]
            normal_vector_component_i = face_rotation_matrix[-1, _i]
        else:
            normal_vector_component_j = -face_rotation_matrix[-1, _j]
            normal_vector_component_i = -face_rotation_matrix[-1, _i]
        m_mas_f = np.zeros((_ck, _cl), dtype=real)
        m_hyb_f = np.zeros((_ck, _fk), dtype=real)
        for qf in range(_f_is):
            x_q_f = face_quadrature_points[:, qf]
            # x_q_f_prime = face.mapping_matrix @ face.quadrature_points[:, qf]
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            phi_k = finite_element.cell_basis_k.evaluate_function(x_q_f, x_c, h_c)
            phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, h_c)
            # phi_k = cell_basis_k.get_phi_vector(x_q_f_prime, x_c, h_c)
            # phi_l = cell_basis_l.get_phi_vector(x_q_f_prime, x_c, h_c)
            psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, h_f)
            m_mas_f += w_q_f * np.tensordot(phi_k, phi_l, axes=0)
            m_hyb_f += w_q_f * np.tensordot(phi_k, psi_k, axes=0)
        _c0 = _i * _cl
        _c1 = (_i + 1) * _cl
        local_grad_matric[:, _c0:_c1] -= (1.0 / 2.0) * m_mas_f * normal_vector_component_j
        _c0 = _dx * _cl + _f * _dx * _fk + _i * _fk
        _c1 = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
        local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * m_hyb_f * normal_vector_component_j
        _c0 = _j * _cl
        _c1 = (_j + 1) * _cl
        local_grad_matric[:, _c0:_c1] -= (1.0 / 2.0) * m_mas_f * normal_vector_component_i
        _c0 = _dx * _cl + _f * _dx * _fk + _j * _fk
        _c1 = _dx * _cl + _f * _dx * _fk + (_j + 1) * _fk
        local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * m_hyb_f * normal_vector_component_i
    local_grad_matric2 = m_mas_inv @ local_grad_matric
    return local_grad_matric2


def get_regular_gradient_component_matrix(
    field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], _i: int, _j: int
) -> ndarray:
    _d = field.euclidean_dimension
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _ck = finite_element.cell_basis_k.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    _io = finite_element.construction_integration_order
    # --- CELL ENVIRONMENT
    _c_is = cell.get_quadrature_size(_io)
    cell_quadrature_points = cell.get_quadrature_points(_io)
    cell_quadrature_weights = cell.get_quadrature_weights(_io)
    x_c = cell.centroid
    h_c = cell.diameter
    local_grad_matric = np.zeros((_ck, _es), dtype=real)
    m_mas = np.zeros((_ck, _ck), dtype=real)
    m_adv_j = np.zeros((_ck, _cl), dtype=real)
    for qc in range(_c_is):
        x_q_c = cell_quadrature_points[:, qc]
        w_q_c = cell_quadrature_weights[qc]
        phi_k = finite_element.cell_basis_k.evaluate_function(x_q_c, x_c, h_c)
        d_phi_l_j = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, h_c, _j)
        m_adv_j += w_q_c * np.tensordot(phi_k, d_phi_l_j, axes=0)
        m_mas += w_q_c * np.tensordot(phi_k, phi_k, axes=0)
    m_mas_inv = np.linalg.inv(m_mas)
    _c0 = _i * _cl
    _c1 = (_i + 1) * _cl
    local_grad_matric[:, _c0:_c1] += m_adv_j
    for _f, face in enumerate(faces):
        # --- FACE ENVIRONMENT
        _f_is = face.get_quadrature_size(_io)
        face_quadrature_points = face.get_quadrature_points(_io)
        face_quadrature_weights = face.get_quadrature_weights(_io)
        h_f = face.diameter
        x_f = face.centroid
        face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
        dist_in_face = (face_rotation_matrix @ (face.centroid - cell.centroid))[-1]
        if dist_in_face > 0:
            normal_vector_component_j = face_rotation_matrix[-1, _j]
        else:
            normal_vector_component_j = -face_rotation_matrix[-1, _j]
        m_mas_f = np.zeros((_ck, _cl), dtype=real)
        m_hyb_f = np.zeros((_ck, _fk), dtype=real)
        for qf in range(_f_is):
            x_q_f = face_quadrature_points[:, qf]
            # x_q_f_prime = face.mapping_matrix @ face.quadrature_points[:, qf]
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            phi_k = finite_element.cell_basis_k.evaluate_function(x_q_f, x_c, h_c)
            phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, h_c)
            # phi_k = cell_basis_k.get_phi_vector(x_q_f_prime, x_c, h_c)
            # phi_l = cell_basis_l.get_phi_vector(x_q_f_prime, x_c, h_c)
            psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, h_f)
            m_mas_f += w_q_f * np.tensordot(phi_k, phi_l, axes=0)
            m_hyb_f += w_q_f * np.tensordot(phi_k, psi_k, axes=0)
        _c0 = _i * _cl
        _c1 = (_i + 1) * _cl
        local_grad_matric[:, _c0:_c1] -= m_mas_f * normal_vector_component_j
        _c0 = _dx * _cl + _f * _dx * _fk + _i * _fk
        _c1 = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
        local_grad_matric[:, _c0:_c1] += m_hyb_f * normal_vector_component_j
    local_grad_matric2 = m_mas_inv @ local_grad_matric
    return local_grad_matric2


def get_gradient_operators(field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape]) -> ndarray:
    _gs = field.gradient_dimension
    _io = finite_element.construction_integration_order
    _c_is = cell.get_quadrature_size(_io)
    cell_quadrature_points = cell.get_quadrature_points(_io)
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _ck = finite_element.cell_basis_k.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    gradient_operators = np.zeros((_c_is, _gs, _es), dtype=real)
    h_c = cell.diameter
    x_c = cell.centroid
    for _qc in range(_c_is):
        x_qc = cell_quadrature_points[:, _qc]
        v_ck = finite_element.cell_basis_k.evaluate_function(x_qc, x_c, h_c)
        for key, val in field.voigt_data.items():
            _i = key[0]
            _j = key[1]
            voigt_indx = val[0]
            voigt_coef = val[1]
            if field.derivation_type == DerivationType.REGULAR:
                gradient_component_matrix = get_regular_gradient_component_matrix(
                    field=field, finite_element=finite_element, cell=cell, faces=faces, _i=_i, _j=_j
                )
            elif field.derivation_type == DerivationType.SYMMETRIC:
                gradient_component_matrix = get_symmetric_gradient_component_matrix(
                    field=field, finite_element=finite_element, cell=cell, faces=faces, _i=_i, _j=_j
                )
            else:
                raise KeyError
            gradient_operators[_qc, voigt_indx] = voigt_coef * v_ck @ gradient_component_matrix
    return gradient_operators
