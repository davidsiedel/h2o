from numpy import ndarray
from typing import Callable
from h2o.h2o import *
from h2o.fem.basis.basis import Basis
from h2o.fem.element.cell import Cell
from h2o.fem.element.face import Face


def get_cell_mass_matrix_in_cell(
    cell: Cell, cell_basis_0: Basis, cell_basis_1: Basis, x_q_c: ndarray, w_q_c: float
) -> ndarray:
    """

    Args:
        cell:
        cell_basis_0:
        cell_basis_1:
        x_q_c:
        w_q_c:

    Returns:

    """
    v_c = cell.shape.diameter
    x_c = cell.shape.centroid
    phi_vector_0 = cell_basis_0.evaluate_function(x_q_c, x_c, v_c)
    phi_vector_1 = cell_basis_1.evaluate_function(x_q_c, x_c, v_c)
    m = w_q_c * np.tensordot(phi_vector_0, phi_vector_1, axes=0)
    return m


def get_cell_stiffness_matrix_in_cell(
    cell: Cell, cell_basis_0: Basis, cell_basis_1: Basis, dx: int, x_q_c: ndarray, w_q_c: float,
) -> ndarray:
    """

    Args:
        cell:
        cell_basis_0:
        cell_basis_1:
        dx:
        x_q_c:
        w_q_c:

    Returns:

    """
    v_c = cell.shape.diameter
    x_c = cell.shape.centroid
    d_phi_vector_0 = cell_basis_0.evaluate_derivative(x_q_c, x_c, v_c, dx)
    d_phi_vector_1 = cell_basis_1.evaluate_derivative(x_q_c, x_c, v_c, dx)
    m = w_q_c * np.tensordot(d_phi_vector_0, d_phi_vector_1, axes=0)
    return m


def get_cell_mass_matrix_in_face(
    cell: Cell, cell_basis_0: Basis, cell_basis_1: Basis, x_q_f: ndarray, w_q_f: float,
) -> ndarray:
    """

    Args:
        cell:
        cell_basis_0:
        cell_basis_1:
        x_q_f:
        w_q_f:

    Returns:

    """
    x_c = cell.shape.centroid
    v_c = cell.shape.diameter
    phi_vector_0 = cell_basis_0.evaluate_function(x_q_f, x_c, v_c)
    phi_vector_1 = cell_basis_1.evaluate_function(x_q_f, x_c, v_c)
    m = w_q_f * np.tensordot(phi_vector_0, phi_vector_1, axes=0)
    return m


def get_cell_advection_matrix_in_cell(
    cell: Cell, cell_basis_0: Basis, cell_basis_1: Basis, dx: int, x_q_c: ndarray, w_q_c: float,
) -> ndarray:
    """

    Args:
        cell:
        cell_basis_0:
        cell_basis_1:
        dx:
        x_q_c:
        w_q_c:

    Returns:

    """
    v_c = cell.shape.diameter
    x_c = cell.shape.centroid
    phi_vector_0 = cell_basis_0.evaluate_function(x_q_c, x_c, v_c)
    d_phi_vector_1 = cell_basis_1.evaluate_derivative(x_q_c, x_c, v_c, dx)
    m = w_q_c * np.tensordot(phi_vector_0, d_phi_vector_1, axes=0)
    return m


def get_cell_advection_matrix_in_face(
    cell: Cell, cell_basis_0: Basis, cell_basis_1: Basis, dx: int, x_q_f: ndarray, w_q_f: float,
) -> ndarray:
    """

    Args:
        cell:
        cell_basis_0:
        cell_basis_1:
        dx:
        x_q_f:
        w_q_f:

    Returns:

    """
    x_c = cell.shape.centroid
    v_c = cell.shape.diameter
    phi_vector_0 = cell_basis_0.evaluate_function(x_q_f, x_c, v_c)
    d_phi_vector_1 = cell_basis_1.evaluate_derivative(x_q_f, x_c, v_c, dx)
    m = w_q_f * np.tensordot(phi_vector_0, d_phi_vector_1, axes=0)
    return m


def get_hybrid_mass_matrix_in_face(
    cell: Cell, face: Face, cell_basis: Basis, face_basis: Basis, x_q_f: ndarray, w_q_f: float,
) -> ndarray:
    """

    Args:
        cell:
        face:
        cell_basis:
        face_basis:
        x_q_f:
        w_q_f:

    Returns:

    """
    v_f = face.shape.diameter
    x_f = face.shape.centroid
    v_c = cell.shape.diameter
    x_c = cell.shape.centroid
    phi_vector = cell_basis.evaluate_function(x_q_f, x_c, v_c)
    s_q_f = (face.mapping_matrix @ x_q_f)[:-1]
    s_f = (face.mapping_matrix @ x_f)[:-1]
    psi_vector = face_basis.evaluate_function(s_q_f, s_f, v_f)
    m = w_q_f * np.tensordot(phi_vector, psi_vector, axes=0)
    return m


def get_test_mass_matrix_in_face(
    cell: Cell, face: Face, cell_basis: Basis, face_basis: Basis, x_q_f: ndarray, w_q_f: float,
) -> ndarray:
    """

    Args:
        cell:
        face:
        cell_basis:
        face_basis:
        x_q_f:
        w_q_f:

    Returns:

    """
    v_f = face.shape.diameter
    x_f = face.shape.centroid
    v_c = cell.shape.diameter
    x_c = cell.shape.centroid
    phi_vector = cell_basis.evaluate_function(x_q_f, x_c, v_c)
    s_q_f = (face.mapping_matrix @ x_q_f)[:-1]
    s_f = (face.mapping_matrix @ x_f)[:-1]
    psi_vector = face_basis.evaluate_function(s_q_f, s_f, v_f)
    m = w_q_f * np.tensordot(psi_vector, phi_vector, axes=0)
    return m


def get_hybrid_advection_matrix_in_face(
    cell: Cell, face: Face, cell_basis: Basis, face_basis: Basis, dx: int, x_q_f: ndarray, w_q_f: float,
) -> ndarray:
    """

    Args:
        cell:
        face:
        cell_basis:
        face_basis:
        dx:
        x_q_f:
        w_q_f:

    Returns:

    """
    v_f = face.shape.diameter
    x_f = face.shape.centroid
    v_c = cell.shape.diameter
    x_c = cell.shape.centroid
    s_f = (face.mapping_matrix @ x_f)[:-1]
    s_q_f = (face.mapping_matrix @ x_q_f)[:-1]
    d_phi_vector = cell_basis.evaluate_derivative(x_q_f, x_c, v_c, dx)
    psi_vector = face_basis.evaluate_function(s_q_f, s_f, v_f)
    m = w_q_f * np.tensordot(d_phi_vector, psi_vector, axes=0)
    return m


def get_face_mass_matrix_in_face(
    face: Face, face_basis_0: Basis, face_basis_1: Basis, x_q_f: ndarray, w_q_f: float
) -> ndarray:
    """

    Args:
        face:
        face_basis_0:
        face_basis_1:
        x_q_f:
        w_q_f:

    Returns:

    """
    v_f = face.shape.diameter
    x_f = face.shape.centroid
    s_q_f = (face.mapping_matrix @ x_q_f)[:-1]
    s_f = (face.mapping_matrix @ x_f)[:-1]
    psi_vector_0 = face_basis_0.evaluate_function(s_q_f, s_f, v_f)
    psi_vector_1 = face_basis_1.evaluate_function(s_q_f, s_f, v_f)
    m = w_q_f * np.tensordot(psi_vector_0, psi_vector_1, axes=0)
    return m


def get_face_pressure_vector_in_face(
    face: Face, face_basis: Basis, pressure: Callable, x_q_f: ndarray, w_q_f: float
) -> ndarray:
    """

    Args:
        face:
        face_basis:
        pressure:
        x_q_f:
        w_q_f:

    Returns:

    """
    v_f = face.shape.diameter
    x_f = face.shape.centroid
    s_f = (face.mapping_matrix @ x_f)[:-1]
    s_q_f = (face.mapping_matrix @ x_q_f)[:-1]
    psi_vector = face_basis.evaluate_function(s_q_f, s_f, v_f)
    v = w_q_f * psi_vector * pressure(s_q_f)
    return v


def get_cell_load_vector_in_cell(
    cell: Cell, cell_basis: Basis, load: Callable, x_q_c: ndarray, w_q_c: float
) -> ndarray:
    """

    Args:
        cell:
        cell_basis:
        load:
        x_q_c:
        w_q_c:

    Returns:

    """
    v_c = cell.shape.diameter
    x_c = cell.shape.centroid
    phi_vector = cell_basis.evaluate_function(x_q_c, x_c, v_c)
    v = w_q_c * phi_vector * load(x_q_c)
    return v


def get_cell_basis_vector(cell: Cell, cell_basis: Basis, x: ndarray) -> ndarray:
    """

    Args:
        cell:
        cell_basis:
        x:

    Returns:

    """
    v_c = cell.shape.diameter
    x_c = cell.shape.centroid
    phi_vector = cell_basis.evaluate_function(x, x_c, v_c)
    return phi_vector




# v_f = face.diameter
# x_f = face.centroid
# # --------------------------------------------------------------------------------------------------------------
# face_mass_matrix_in_face = np.zeros((face_basis_0.basis_dimension, face_basis_1.basis_dimension))
# # --------------------------------------------------------------------------------------------------------------
# x_f_in_face = Face.get_points_in_face_reference_frame(face.centroid, face_reference_frame_transformation_matrix)
# face_quadrature_points_in_face = Face.get_points_in_face_reference_frame(
#     face.quadrature_points, face_reference_frame_transformation_matrix
# )
# for x_Q_f_in_face, w_Q_f in zip(face_quadrature_points_in_face, face.quadrature_weights):
#     # ----------------------------------------------------------------------------------------------------------
#     psi_vector_0 = face_basis_0.get_phi_vector(x_Q_f_in_face, x_f_in_face, v_f)
#     number_of_components = psi_vector_0.shape[0]
#     psi_vector_0 = np.resize(psi_vector_0, (1, number_of_components))
#     # ----------------------------------------------------------------------------------------------------------
#     psi_vector_1 = face_basis_1.get_phi_vector(x_Q_f_in_face, x_f_in_face, v_f)
#     number_of_components = psi_vector_1.shape[0]
#     psi_vector_1 = np.resize(psi_vector_1, (1, number_of_components))
#     # ----------------------------------------------------------------------------------------------------------
#     m = w_Q_f * psi_vector_0.T @ psi_vector_1
#     face_mass_matrix_in_face += m
