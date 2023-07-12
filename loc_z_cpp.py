# from unittest import TestCase

from h2o.problem.problem import Problem
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.material import Material
from h2o.problem.load import Load
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
from h2o.h2o import *
from mgis import behaviour as mgis_bv
from h2o.geometry.shape import Shape, get_rotation_matrix
from h2o.fem.element.operators.gradient import get_symmetric_cartesian_gradient_component_matrix_rhs
from h2o.geometry.shape import Shape
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
import h2o.fem.element.operators.gradient as gradient_operator
import h2o.fem.element.operators.stabilization as stabilization_operator
import h2o.fem.element.operators.identity as identity_operator
from h2o.problem.material import Material
from scipy.sparse import coo_matrix

np.set_printoptions(precision=2)

# from h2o.problem.solve.solve_implicit_new import solve_implicit
# from h2o.problem.solve.solve_condensation import solve_condensation
from h2o.problem.solve.solve_cpp2 import solve_condensation as solve

# --- VALUES
u_min = 0.0
u_max = 1.0e-2
steps = 20
time_steps = np.linspace(u_min, u_max, steps)
iterations = 10
print(list(time_steps))

# --- LOAD
def volumetric_load(time: float, position: ndarray):
    return 0

loads = [Load(volumetric_load, 0), Load(volumetric_load, 1)]

# --- BC
def pull(time: float, position: ndarray) -> float:
    return time

def fixed(time: float, position: ndarray) -> float:
    return 0.0

boundary_conditions = [
    # BoundaryCondition("LRU", pull, BoundaryType.DISPLACEMENT, 1),
    # BoundaryCondition("LRD", fixed, BoundaryType.DISPLACEMENT, 1),
    # BoundaryCondition("AXESYM", fixed, BoundaryType.DISPLACEMENT, 0),
    BoundaryCondition("TOP", pull, BoundaryType.DISPLACEMENT, 1),
    BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 1),
    BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
]

# --- MESH
# mesh_file_path = "meshes/ssna.geof"
# mesh_file_path = "z_cpp/quadrangle050.msh"
# mesh_file_path = "z_cpp/triangle_structured_00001.msh"
mesh_file_path = "/home/dsiedel/projetcs/lolita/applications/linear_elastic_tensile_square/mesh2.msh"
# mesh_file_path = "meshes/ssna_quad_mid.msh"
# mesh_file_path = "meshes/ssna303_COMP_AXI_on_axis.msh"

# --- FIELD
displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN)

# --- FINITE ELEMENT
finite_element = FiniteElement(
    element_type=ElementType.HHO_EQUAL,
    polynomial_order=1,
    euclidean_dimension=displacement.euclidean_dimension,
    basis_type=BasisType.MONOMIAL,
)


start_time = time.time()

# --- PROBLEM
problem = Problem(
    mesh_file_path=mesh_file_path,
    field=displacement,
    finite_element=finite_element,
    time_steps=time_steps,
    iterations=iterations,
    boundary_conditions=boundary_conditions,
    loads=loads,
    quadrature_type=QuadratureType.GAUSS,
    tolerance=1.0e-6,
    res_folder_path= "/home/dsiedel/projetcs/h2o/z_cpp/out"
)

# elem_1 = problem.elements[4]
# klmp = gradient_operator.get_symmetric_cartesian_gradient_component_matrix_rhs(elem_1.field, finite_element, elem_1.cell, elem_1.faces, 0, 0)

print("build time : --- %s seconds ---" % (time.time() - start_time))

# --- MATERIAL
parameters = {"YoungModulus": 206.9e9, "PoissonRatio": 0.2}
stabilization_parameter = parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
mat = Material(
    nq=problem.mesh.number_of_cell_quadrature_points_in_mesh,
    library_path="/home/dsiedel/projetcs/h2o/behaviours/bhv_isotropic_linear_hardening/src/libBehaviour.so",
    library_name="IsotropicLinearHardeningPlasticity",
    # library_name="FiniteStrainIsotropicLinearHardeningPlasticity",
    hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
    stabilization_parameter=stabilization_parameter,
    lagrange_parameter=parameters["YoungModulus"],
    field=displacement,
    parameters=None,
)

# elem_list = [
#     problem.elements[0],
#     problem.elements[4],
#     problem.elements[1],
#     problem.elements[5],
#     problem.elements[2],
#     problem.elements[6],
#     problem.elements[7],
#     problem.elements[3],
# ]
# for ielem, element in enumerate(problem.elements):
#     print("****** element :")
#     print(mat2str(element.cell.vertices))
#     for f_count, face in enumerate(element.faces):
#         dist_in_face = (face.get_rotation_matrix() @ (face.get_centroid() - element.cell.get_centroid()))[-1]
#         sign  = 1
#         if dist_in_face < 0:
#             sign = -1
#         print("-- face :", f_count)
#         print(mat2str(face.vertices))
#         print("-- face_orientation")
#         print(sign)
#         print("-- face_orientation")
#         print(mat2str(face.get_rotation_matrix()))

#     print("-- stabilization :")
#     print(mat2str(element.stabilization_operator))
#     _io: int = problem.finite_element.computation_integration_order
#     # klmp = gradient_operator.get_symmetric_cartesian_gradient_component_matrix_rhs(element.field, finite_element, element.cell, element.faces, 0, 1)
#     for i_grad, grad in enumerate(element.gradients_operators):
#         print("-- grad", i_grad)
#         print(mat2str(grad))

solve(
    problem,
    mat,
    verbose=False,
    debug_mode=DebugMode.NONE,
    accelerate=0,
    num_local_iterations=0
)

# for element in [problem.elements[1]]:
# for element in problem.elements:
# for ielem, element in enumerate([problem.elements[1], problem.elements[0], problem.elements[2]]):
#     get_symmetric_cartesian_gradient_component_matrix_rhs(displacement, finite_element, element.cell, element.faces, 0, 0)

# for ielem, element in enumerate([problem.elements[1], problem.elements[0], problem.elements[2]]):
#     # --------------------------------------------------------------------------------------------------
#     # INTEGRATION
#     # --------------------------------------------------------------------------------------------------
#     _io: int = problem.finite_element.computation_integration_order
#     cell_quadrature_size = element.cell.get_quadrature_size(_io, quadrature_type=problem.quadrature_type)
#     cell_quadrature_points = element.cell.get_quadrature_points(_io, quadrature_type=problem.quadrature_type)
#     cell_quadrature_weights = element.cell.get_quadrature_weights(_io, quadrature_type=problem.quadrature_type)
#     x_c: ndarray = element.cell.get_centroid()
#     bdc: ndarray = element.cell.get_bounding_box()
#     print("*********************** finite_element :")
#     print(element.cell.vertices)
#     #
#     _i = 0
#     _j = 0
#     _io2: int = problem.finite_element.construction_integration_order
#     cell_quadrature_size2 = element.cell.get_quadrature_size(_io2, quadrature_type=problem.quadrature_type)
#     cell_quadrature_points2 = element.cell.get_quadrature_points(_io2, quadrature_type=problem.quadrature_type)
#     cell_quadrature_weights2 = element.cell.get_quadrature_weights(_io2, quadrature_type=problem.quadrature_type)
#     cell_row_vectors = np.zeros((3, cell_quadrature_size2))
#     cell_colI_vectors = np.zeros((3, cell_quadrature_size2))
#     cell_colJ_vectors = np.zeros((3, cell_quadrature_size2))
#     wts = np.zeros((cell_quadrature_size2, ))
#     for _qc in range(cell_quadrature_size2):
#         _w_q_c = cell_quadrature_weights2[_qc]
#         _x_q_c = cell_quadrature_points2[:, _qc]
#         phi_k = finite_element.cell_basis_k.evaluate_function(_x_q_c, x_c, bdc)
#         d_phi_l_j = finite_element.cell_basis_l.evaluate_derivative(_x_q_c, x_c, bdc, _j)
#         d_phi_l_i = finite_element.cell_basis_l.evaluate_derivative(_x_q_c, x_c, bdc, _i)
#         cell_row_vectors[:, _qc] = phi_k
#         cell_colI_vectors[:, _qc] = d_phi_l_i
#         cell_colJ_vectors[:, _qc] = d_phi_l_j
#         wts[_qc] = _w_q_c
#         # print("* _w_q_c ", _qc, ":")
#         # print(_w_q_c)
#         # print("* _x_q_c ", _qc, ":")
#         # print(_x_q_c)
#         # print("* B ", _qc, ":")
#         # print(element.gradients_operators[_qc])
#     #
#     # print("* cell_row_vectors :")
#     # print(cell_row_vectors)
#     # print("* cell_colI_vectors :")
#     # print(cell_colI_vectors)
#     # print("* cell_colJ_vectors :")
#     # print(cell_colJ_vectors)
#     # print("* wts :")
#     # print(wts)
#     element_stiffness_matrix = np.zeros((element.element_size, element.element_size), dtype=real)
#     element_internal_forces = np.zeros((element.element_size,), dtype=real)
#     element_external_forces = np.zeros((element.element_size,), dtype=real)
#     # if ielem == 1:
#     #     get_symmetric_cartesian_gradient_component_matrix_rhs(displacement, finite_element, element.cell, element.faces, 0, 0)
#     print("* op (0, 0) :")
#     print(get_symmetric_cartesian_gradient_component_matrix_rhs(displacement, finite_element, element.cell, element.faces, 0, 0))
#     print("* op (0, 1) :")
#     print(get_symmetric_cartesian_gradient_component_matrix_rhs(displacement, finite_element, element.cell, element.faces, 0, 1))
#     print("* op (1, 0) :")
#     print(get_symmetric_cartesian_gradient_component_matrix_rhs(displacement, finite_element, element.cell, element.faces, 1, 0))
#     print("* op (1, 1) :")
#     print(get_symmetric_cartesian_gradient_component_matrix_rhs(displacement, finite_element, element.cell, element.faces, 1, 1))
#     for _qc in range(cell_quadrature_size):
#         _qp = element.quad_p_indices[_qc]
#         _w_q_c = cell_quadrature_weights[_qc]
#         _x_q_c = cell_quadrature_points[:, _qc]
#         print("* _w_q_c ", _qc, ":")
#         print(_w_q_c)
#         print("* _x_q_c ", _qc, ":")
#         print(_x_q_c)
#         print("* B ", _qc, ":")
#         print(element.gradients_operators[_qc])
#     for if_, face in enumerate(element.faces):
#         print("*** face :")
#         print(face.vertices)
#         # --- FACE GEOMETRY
#         x_f = face.get_centroid()
#         # print("* x_f :")
#         # print(face.get_centroid())
#         bdf_proj = face.get_face_bounding_box()
#         face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
#         dist_in_face = (face_rotation_matrix @ (x_f - x_c))[-1]
#         _j = 0
#         _i = 0
#         sign = 1.0
#         if dist_in_face > 0:
#             normal_vector_component_j = sign = +1.0
#             normal_vector_component_i = sign = +1.0
#         else:
#             normal_vector_component_j = sign = -1.0 #-face_rotation_matrix[-1, _j]
#             normal_vector_component_i = sign = -1.0 #-face_rotation_matrix[-1, _i]
#         print("* sign :")
#         print(sign)
#         print("* normal_vector :")
#         print(face_rotation_matrix[-1,:])
#         _io = finite_element.construction_integration_order
#         # _io = 2
#         _f_is = face.get_quadrature_size(_io)
#         face_quadrature_points = face.get_quadrature_points(_io)
#         face_quadrature_weights = face.get_quadrature_weights(_io)
#         mat = np.zeros((2, _io))
#         wts = np.zeros((_io, ))
#         x_q_fs = np.zeros((2, _io))
#         for qf in range(_f_is):
#             x_q_f = face_quadrature_points[:, qf]
#             w_q_f = face_quadrature_weights[qf]
#             s_f = (face_rotation_matrix @ x_f)[:-1]
#             s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
#             phi_k = finite_element.cell_basis_k.evaluate_function(x_q_f, x_c, bdc)
#             phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
#             psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
#             mat[:, qf] = psi_k
#             wts[qf] = w_q_f
#             x_q_fs[:, qf] = x_q_f
#         print("* col_face_vector :")
#         print(mat)
#         print("* wts :")
#         print(wts)
#         print("* ips :")
#         print(x_q_fs)


# # --- SOLVE
# if algorithm_type == "STATIC":
#     solve_condensation(p, mat, verbose=False, debug_mode=DebugMode.NONE)
# elif algorithm_type == "IMPLICIT":
#     solve_implicit(p, mat, verbose=False, debug_mode=DebugMode.NONE)
