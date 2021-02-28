from typing import List

from h2o.fem.element.elements.blocks import *
from h2o.fem.element.finite_element import FiniteElement
from h2o.pbbb.field import Field

def get_projection_in_cell(field: Field, finite_element: FiniteElement, cell: Cell, faces: List[Face], functions: List[Callable]) -> ndarray:
    _d = field.euclidean_dimension
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _ck = finite_element.cell_basis_k.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)

    return

def get_projection_in_element(
    field: Field, finite_element: FiniteElement, cell: Cell, faces: List[Face], functions: List[Callable]
) -> ndarray:
    """

    Args:
        field:
        finite_element:
        cell:
        faces:
        functions:

    Returns:

    """
    _d = field.euclidean_dimension
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _ck = finite_element.cell_basis_k.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    # _gs = field.gradient_dimension
    # --- MATRIX PART
    interpolation_matrix = np.zeros((_es, _es), dtype=real)
    m_cell_mass = np.zeros((_cl, _cl), dtype=real)
    for _qc in range(len(cell.quadrature_weights)):
        _x_qc = cell.quadrature_points[:, _qc]
        _w_qc = cell.quadrature_weights[_qc]
        m_cell_mass += get_cell_mass_matrix_in_cell(
            cell=cell,
            cell_basis_0=finite_element.cell_basis_l,
            cell_basis_1=finite_element.cell_basis_l,
            x_q_c=_x_qc,
            w_q_c=_w_qc,
        )
    for _x_dir in range(_dx):
        _i_0 = _x_dir * _cl
        _i_1 = (_x_dir + 1) * _cl
        interpolation_matrix[_i_0:_i_1, _i_0:_i_1] += m_cell_mass
    for _f in range(_nf):
        m_face_mass = np.zeros((_fk, _fk), dtype=real)
        for _qf in range(len(faces[_f].quadrature_weights)):
            _x_qf = cell.quadrature_points[:, _qf]
            _w_qf = cell.quadrature_weights[_qf]
            m_face_mass += get_face_mass_matrix_in_face(
                faces[_f], finite_element.face_basis_k, finite_element.face_basis_k, _x_qf, _w_qf
            )
        for _x_dir in range(_dx):
            _i_0 = _dx * _cl + _f * _dx * _fk + _x_dir * _fk
            _i_1 = _dx * _cl + _f * _dx * _fk + (_x_dir + 1) * _fk
            interpolation_matrix[_i_0:_i_1, _i_0:_i_1] += m_face_mass
    # --- VECTOR PART
    interpolation_vector = np.zeros((_es,), dtype=real)
    for _qc in range(len(cell.quadrature_weights)):
        _x_qc = cell.quadrature_points[:, _qc]
        _w_qc = cell.quadrature_weights[_qc]
        v = finite_element.cell_basis_l.evaluate_function(_x_qc, cell.shape.centroid, cell.shape.diameter)
        for _x_dir in range(_dx):
            _i_0 = _x_dir * _cl
            _i_1 = (_x_dir + 1) * _cl
            interpolation_vector[_i_0:_i_1] += _w_qc * v * functions[_x_dir](_x_qc)
    for _f in range(_nf):
        for _qf in range(len(faces[_f].quadrature_weights)):
            _x_qf = cell.quadrature_points[:, _qf]
            _w_qf = cell.quadrature_weights[_qf]
            _s_qf = (faces[_f].mapping_matrix @ _x_qf)[:-1]
            _s_f = (faces[_f].mapping_matrix @ faces[_f].shape.centroid)[:-1]
            v = finite_element.face_basis_k.evaluate_function(_s_qf, _s_f, faces[_f].shape.diameter)
            for _x_dir in range(_dx):
                _i_0 = _dx * _cl + _f * _fk + _x_dir * _fk
                _i_1 = _dx * _cl + _f * _fk + (_x_dir + 1) * _fk
                interpolation_vector[_i_0:_i_1] += _w_qf * v * functions[_x_dir](_x_qf)
    projection_vector = np.linalg.solve(interpolation_matrix, interpolation_vector)
    return projection_vector