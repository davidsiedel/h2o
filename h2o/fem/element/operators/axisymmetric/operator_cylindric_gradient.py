import numpy as np

from h2o.h2o import *
from h2o.fem.element.finite_element import FiniteElement
from h2o.geometry.shape import Shape, get_rotation_matrix
from h2o.field.field import Field
import sys


def get_angular_reconstruction_component_matrix(
        field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], _i: int
) -> ndarray:
    # --- ELEMENT DATA
    _d = field.euclidean_dimension
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _cr = finite_element.cell_basis_r.dimension
    _cr_star = finite_element.cell_basis_r.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    # --- CELL GEOMETRY
    x_c = cell.get_centroid()
    bdc = cell.get_bounding_box()
    # --- CELL ENVIRONMENT
    _io = finite_element.construction_integration_order
    _c_is = cell.get_quadrature_size(_io)
    cell_quadrature_points = cell.get_quadrature_points(_io)
    cell_quadrature_weights = cell.get_quadrature_weights(_io)
    potential_lhs: ndarray = np.zeros((_cr, _cr), dtype=real)
    # ------------------------------------------------------------------------------------------------------------------
    # LHS
    # ------------------------------------------------------------------------------------------------------------------
    for qc in range(_c_is):
        x_q_c: ndarray = cell_quadrature_points[:, qc]
        w_q_c: float = cell_quadrature_weights[qc]
        for _j in range(2):
            d_phi_r_j: ndarray = finite_element.cell_basis_r.evaluate_derivative(x_q_c, x_c, bdc, _j)
            potential_lhs += w_q_c * np.tensordot(d_phi_r_j, d_phi_r_j, axes=0)
        phi_r: ndarray = finite_element.cell_basis_r.evaluate_function(x_q_c, x_c, bdc)
        potential_lhs += (1.0 / (x_q_c[0] ** 2)) * w_q_c * np.tensordot(phi_r, phi_r, axes=0)
    # potential_lhs_inv: ndarray = np.linalg.inv(potential_lhs[1:, 1:])
    potential_lhs_inv: ndarray = np.linalg.inv(potential_lhs)
    # ------------------------------------------------------------------------------------------------------------------
    # RHS
    # ------------------------------------------------------------------------------------------------------------------
    potential_rhs = np.zeros((_cr, _es), dtype=real)
    for qc in range(_c_is):
        x_q_c: ndarray = cell_quadrature_points[:, qc]
        w_q_c: float = cell_quadrature_weights[qc]
        for _j in range(2):
            d_phi_r_j: ndarray = finite_element.cell_basis_r.evaluate_derivative(x_q_c, x_c, bdc, _j)
            d_phi_l_j: ndarray = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, bdc, _j)
            _c0 = _i * _cl
            _c1 = (_i + 1) * _cl
            potential_rhs[:, _c0:_c1] += w_q_c * np.tensordot(d_phi_r_j, d_phi_l_j, axes=0)
        phi_r: ndarray = finite_element.cell_basis_r.evaluate_function(x_q_c, x_c, bdc)
        phi_l: ndarray = finite_element.cell_basis_l.evaluate_function(x_q_c, x_c, bdc)
        _c0 = _i * _cl
        _c1 = (_i + 1) * _cl
        potential_rhs[:, _c0:_c1] += (1.0 / (x_q_c[0] ** 2)) * w_q_c * np.tensordot(phi_r, phi_l, axes=0)
    for _f, face in enumerate(faces):
        x_f = face.get_centroid()
        bdf_proj = face.get_face_bounding_box()
        face_rotation_matrix = face.get_rotation_matrix().copy()
        dist_in_face = (face_rotation_matrix @ (x_f - x_c))[-1]
        _f_is = face.get_quadrature_size(_io)
        face_quadrature_points = face.get_quadrature_points(_io)
        face_quadrature_weights = face.get_quadrature_weights(_io)
        for qf in range(_f_is):
            x_q_f = face_quadrature_points[:, qf]
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            for _j in range(2):
                if dist_in_face > 0:
                    normal_vector_component_j = face_rotation_matrix[-1, _j]
                else:
                    normal_vector_component_j = -face_rotation_matrix[-1, _j]
                d_phi_r_j = finite_element.cell_basis_r.evaluate_derivative(x_q_f, x_c, bdc, _j)
                phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
                psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
                _c0 = _i * _cl
                _c1 = (_i + 1) * _cl
                potential_rhs[:, _c0:_c1] -= w_q_f * np.tensordot(d_phi_r_j, phi_l, axes=0) * normal_vector_component_j
                _c0 = _dx * _cl + _f * _dx * _fk + _i * _fk
                _c1 = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
                potential_rhs[:, _c0:_c1] += w_q_f * np.tensordot(d_phi_r_j, psi_k, axes=0) * normal_vector_component_j
    potential_operator: ndarray = np.zeros((_cr, _es), dtype=real)
    # local_recons_matric2[1:, :] = potential_lhs_inv @ potential_rhs[1:, :]
    potential_operator = potential_lhs_inv @ potential_rhs
    # ------------------------------------------------------------------------------------------------------------------
    # CONST
    # ------------------------------------------------------------------------------------------------------------------
    # const_lhs: real = 0.0
    # const_rhs: ndarray = np.zeros((_es,), dtype=real)
    # for qc in range(_c_is):
    #     x_q_c: ndarray = cell_quadrature_points[:, qc]
    #     w_q_c: float = cell_quadrature_weights[qc]
    #     phi_r: ndarray = finite_element.cell_basis_r.evaluate_function(x_q_c, x_c, bdc)
    #     phi_l: ndarray = finite_element.cell_basis_l.evaluate_function(x_q_c, x_c, bdc)
    #     const_rhs -= w_q_c * phi_r[1:] @ potential_operator[1:, :]
    #     _c0 = _i * _cl
    #     _c1 = (_i + 1) * _cl
    #     const_rhs[_c0:_c1] += w_q_c * phi_l
    #     const_lhs += w_q_c
    # potential_operator[1, :] = (1.0 / const_lhs) * const_rhs
    return potential_operator


def get_regular_reconstruction_component_matrix(
        field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], _i: int
) -> ndarray:
    # --- ELEMENT DATA
    _d = field.euclidean_dimension
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _cr = finite_element.cell_basis_r.dimension
    _cr_star = finite_element.cell_basis_r.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    # --- CELL GEOMETRY
    x_c = cell.get_centroid()
    bdc = cell.get_bounding_box()
    # --- CELL ENVIRONMENT
    _io = finite_element.construction_integration_order
    _c_is = cell.get_quadrature_size(_io)
    cell_quadrature_points = cell.get_quadrature_points(_io)
    cell_quadrature_weights = cell.get_quadrature_weights(_io)
    potential_lhs: ndarray = np.zeros((_cr, _cr), dtype=real)
    # ------------------------------------------------------------------------------------------------------------------
    # LHS
    # ------------------------------------------------------------------------------------------------------------------
    for qc in range(_c_is):
        x_q_c: ndarray = cell_quadrature_points[:, qc]
        w_q_c: float = cell_quadrature_weights[qc]
        for _j in range(_dx):
            d_phi_r_j: ndarray = finite_element.cell_basis_r.evaluate_derivative(x_q_c, x_c, bdc, _j)
            potential_lhs += w_q_c * np.tensordot(d_phi_r_j, d_phi_r_j, axes=0)
    potential_lhs_inv: ndarray = np.linalg.inv(potential_lhs[1:, 1:])
    # potential_lhs_inv: ndarray = np.linalg.inv(potential_lhs)
    # ------------------------------------------------------------------------------------------------------------------
    # RHS
    # ------------------------------------------------------------------------------------------------------------------
    potential_rhs = np.zeros((_cr, _es), dtype=real)
    for qc in range(_c_is):
        x_q_c: ndarray = cell_quadrature_points[:, qc]
        w_q_c: float = cell_quadrature_weights[qc]
        for _j in range(_dx):
            d_phi_r_j: ndarray = finite_element.cell_basis_r.evaluate_derivative(x_q_c, x_c, bdc, _j)
            d_phi_l_j: ndarray = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, bdc, _j)
            _c0 = _i * _cl
            _c1 = (_i + 1) * _cl
            potential_rhs[:, _c0:_c1] += w_q_c * np.tensordot(d_phi_r_j, d_phi_l_j, axes=0)
    for _f, face in enumerate(faces):
        x_f = face.get_centroid()
        bdf_proj = face.get_face_bounding_box()
        face_rotation_matrix = face.get_rotation_matrix().copy()
        dist_in_face = (face_rotation_matrix @ (x_f - x_c))[-1]
        _f_is = face.get_quadrature_size(_io)
        face_quadrature_points = face.get_quadrature_points(_io)
        face_quadrature_weights = face.get_quadrature_weights(_io)
        for qf in range(_f_is):
            x_q_f = face_quadrature_points[:, qf]
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            for _j in range(2):
                if dist_in_face > 0:
                    normal_vector_component_j = face_rotation_matrix[-1, _j]
                else:
                    normal_vector_component_j = -face_rotation_matrix[-1, _j]
                d_phi_r_j = finite_element.cell_basis_r.evaluate_derivative(x_q_f, x_c, bdc, _j)
                phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
                psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
                _c0 = _i * _cl
                _c1 = (_i + 1) * _cl
                potential_rhs[:, _c0:_c1] -= w_q_f * np.tensordot(d_phi_r_j, phi_l, axes=0) * normal_vector_component_j
                _c0 = _dx * _cl + _f * _dx * _fk + _i * _fk
                _c1 = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
                potential_rhs[:, _c0:_c1] += w_q_f * np.tensordot(d_phi_r_j, psi_k, axes=0) * normal_vector_component_j
    potential_operator: ndarray = np.zeros((_cr, _es), dtype=real)
    potential_operator[1:, :] = potential_lhs_inv @ potential_rhs[1:, :]
    # potential_operator = potential_lhs_inv @ potential_rhs
    # ------------------------------------------------------------------------------------------------------------------
    # CONST
    # ------------------------------------------------------------------------------------------------------------------
    const_lhs: real = 0.0
    const_rhs: ndarray = np.zeros((_es,), dtype=real)
    for qc in range(_c_is):
        x_q_c: ndarray = cell_quadrature_points[:, qc]
        w_q_c: float = cell_quadrature_weights[qc]
        phi_r: ndarray = finite_element.cell_basis_r.evaluate_function(x_q_c, x_c, bdc)
        phi_l: ndarray = finite_element.cell_basis_l.evaluate_function(x_q_c, x_c, bdc)
        const_rhs -= w_q_c * phi_r[1:] @ potential_operator[1:, :]
        _c0 = _i * _cl
        _c1 = (_i + 1) * _cl
        const_rhs[_c0:_c1] += w_q_c * phi_l
        const_lhs += w_q_c
    potential_operator[1, :] = (1.0 / const_lhs) * const_rhs
    return potential_operator

def get_axisymmetrical_gradient(field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape]):
    dr = get_angular_reconstruction_component_matrix(field, finite_element, cell, faces, 0)
    dz = get_regular_reconstruction_component_matrix(field, finite_element, cell, faces, 1)
    _d = field.euclidean_dimension
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _ck = finite_element.cell_basis_k.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    _gs = field.gradient_dimension
    # --- INTEGRATION ORDER
    _io = finite_element.computation_integration_order
    # _io = finite_element.k_order + finite_element.k_order
    _c_is = cell.get_quadrature_size(_io)
    cell_quadrature_points = cell.get_quadrature_points(_io)
    cell_quadrature_weights = cell.get_quadrature_weights(_io)
    gradient_operators = np.zeros((_c_is, _gs, _es), dtype=real)
    # --- CELL GEOMETRY
    h_c = cell.get_diameter()
    bdc = cell.get_bounding_box()
    x_c = cell.get_centroid()
    for _qc in range(_c_is):
        x_q_c: ndarray = cell_quadrature_points[:, _qc]
        w_q_c: float = cell_quadrature_weights[_qc]
        # v_ck = finite_element.cell_basis_k.evaluate_function(x_qc, x_c, h_c)
        v_ck = finite_element.cell_basis_k.evaluate_function(x_q_c, x_c, bdc)
        lhs: ndarray = w_q_c * np.tensordot(v_ck, v_ck, axes=0)
        lhs_inv = np.linalg.inv(lhs)
        for key, val in field.voigt_data.items():
            _i = key[0]
            _j = key[1]
            voigt_indx = val[0]
            voigt_coef = val[1]
            if _i == 0 and _j != 3:
                d_phi_r_j: ndarray = finite_element.cell_basis_r.evaluate_derivative(x_q_c, x_c, bdc, _j)
                rhs: ndarray = w_q_c * np.tensordot(v_ck, d_phi_r_j, axes=0)
                gradient_component_matrix = lhs_inv @ rhs @ dr
                gradient_operators[_qc, voigt_indx] = voigt_coef * v_ck @ gradient_component_matrix
            elif _i == 1 and _j != 3:
                d_phi_r_j: ndarray = finite_element.cell_basis_r.evaluate_derivative(x_q_c, x_c, bdc, _j)
                rhs: ndarray = w_q_c * np.tensordot(v_ck, d_phi_r_j, axes=0)
                gradient_component_matrix = lhs_inv @ rhs @ dz
                gradient_operators[_qc, voigt_indx] = voigt_coef * v_ck @ gradient_component_matrix
            else:
                phi_r: ndarray = finite_element.cell_basis_r.evaluate_function(x_q_c, x_c, bdc)
                rhs: ndarray = (1.0 / x_q_c[0]) * w_q_c * np.tensordot(v_ck, phi_r, axes=0)
                gradient_component_matrix = lhs_inv @ rhs @ dr
                gradient_operators[_qc, voigt_indx] = voigt_coef * v_ck @ gradient_component_matrix
    return gradient_operators

