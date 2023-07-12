import numpy as np

from h2o.problem.problem import Problem
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.material import Material
from h2o.problem.load import Load
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
from h2o.h2o import *
from h2o.problem.solve.solve_condensation_axi import solve_condensation
from h2o.problem.solve.solve_implicit_axi import solve_implicit
# from h2o.problem.resolution.solve_static_condensation_thermo_axi import solve_newton_static_condensation as solve_axi
from h2o.problem.solve.solve_condensation_thermo_axi import solve_newton_static_condensation as solve_axi

from mgis import behaviour as mgis_bv

# --- VALUES
ts = np.linspace(0.0, 0.1/2.0, 4)
ts = np.linspace(0.0, 1.0, 4)
time_steps = list(ts)
iterations = 10

# --- LOAD
def fr(time: float, position: ndarray):
    return 0.0

def fz(time: float, position: ndarray):
    return 0.0

loads = [Load(fr, 0), Load(fz, 1)]

# --- BC
def clamped(time: float, position: ndarray) -> float:
    return 0.0

boundary_conditions = [
    BoundaryCondition("LEFT", clamped, BoundaryType.DISPLACEMENT, 0),
    # BoundaryCondition("LEFT", fixed_sin, BoundaryType.DISPLACEMENT, 1),
    # BoundaryCondition("RIGHT", fixed_sin, BoundaryType.DISPLACEMENT, 0),
    BoundaryCondition("BOTTOM", clamped, BoundaryType.DISPLACEMENT, 1),
    # BoundaryCondition("RIGHT", fixed_sin, BoundaryType.DISPLACEMENT, 1),
]

# --- MESH
mesh_file_path = "meshes/rod_half.msh"
mesh_file_path = "meshes/rod/satoh_20.msh"
mesh_file_path = "meshes/rod/satoh_10.msh"
# mesh_file_path = "meshes/rod_half_20.msh"
# mesh_file_path = "meshes/satoh_10.msh"
# mesh_file_path = "meshes/unit_square.msh"
# mesh_file_path = "meshes/unit_square_10.msh"

# --- FIELD
displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_SMALL_STRAIN_AXISYMMETRIC)

# --- FINITE ELEMENT
finite_element = FiniteElement(
    element_type=ElementType.HHO_EQUAL,
    polynomial_order=1,
    euclidean_dimension=displacement.euclidean_dimension,
    basis_type=BasisType.MONOMIAL,
)

# --- PROBLEM
p = Problem(
    mesh_file_path=mesh_file_path,
    field=displacement,
    finite_element=finite_element,
    time_steps=time_steps,
    iterations=iterations,
    boundary_conditions=boundary_conditions,
    loads=loads,
    quadrature_type=QuadratureType.GAUSS,
    tolerance=1.0e-8,
    res_folder_path=get_current_res_folder_path() + "_rod"
)

# --- MATERIAL
parameters = {"YoungModulus": 150.0e9, "PoissonRatio": 0.4999}
stabilization_parameter = 1.0 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
mat = Material(
    nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
    library_path="behaviours/bhv_linear_thermo_elasticity/src/libBehaviour.so",
    library_name="LinearThermoElasticity",
    hypothesis=mgis_bv.Hypothesis.AXISYMMETRICAL,
    stabilization_parameter=stabilization_parameter,
    lagrange_parameter=stabilization_parameter,
    field=displacement,
    # integration_type=mgis_bv.IntegrationType.IntegrationWithElasticOperator,
    integration_type=mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator,
    parameters=None,
)

# --- SOLVE
solve_axi(p, mat, verbose=False, debug_mode=DebugMode.NONE)
# solve_cartesian(p, mat, verbose=False, debug_mode=DebugMode.NONE)
