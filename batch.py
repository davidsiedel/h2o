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
from h2o.problem.solve.solve_condensation_axi import solve_condensation as solve

from mgis import behaviour as mgis_bv

batch = "6"

algorithm_rule = {
    "Explicit": 0,
    "Implicit": 50,
}

acceleration_rule = {
    "NonAccelerated" : 0,
    "Accelerated" : 1,
}

jacobian_rule = {
    "Consistent": mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator,
    "Elastic": mgis_bv.IntegrationType.IntegrationWithElasticOperator,
}

def solve_swelling_sphere(computation):

    # --- VALUES
    # time_steps = np.linspace(0.0, 0.20238839836127986 * 5.0, 80, endpoint=True)
    time_steps = np.linspace(0.0, 0.20238839836127986, 40, endpoint=True)
    iterations = 200

    # --- LOAD
    def volumetric_load(time: float, position: ndarray):
        return 0

    loads = [Load(volumetric_load, 0), Load(volumetric_load, 1)]

    # --- BC
    def swell_x(time: float, position: ndarray) -> float:
        return time * position[0]

    def swell_y(time: float, position: ndarray) -> float:
        return time * position[1]

    def clamped(time: float, position: ndarray) -> float:
        return 0.0

    boundary_conditions = [
        BoundaryCondition("TOP", clamped, BoundaryType.DISPLACEMENT, 0),
        BoundaryCondition("BOTTOM", clamped, BoundaryType.DISPLACEMENT, 1),
        BoundaryCondition("INTERIOR", swell_x, BoundaryType.DISPLACEMENT, 0),
        BoundaryCondition("INTERIOR", swell_y, BoundaryType.DISPLACEMENT, 1),
    ]

    # --- MESH
    mesh_file_path = "meshes/sphere/sphere_{}.msh".format(computation["mesh"])

    # --- FIELD
    displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_SMALL_STRAIN_AXISYMMETRIC)

    # --- FINITE ELEMENT
    finite_element = FiniteElement(
        element_type=ElementType.HHO_HIGH,
        polynomial_order=computation["order"],
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
        tolerance=1.0e-6,
        res_folder_path=get_current_res_folder_path() + "_{}_swelling_sphere__{}__hho{}__{}__{}__{}".format(
            batch,
            computation["mesh"],
            computation["order"],
            computation["algorithm"],
            computation["jacobian"],
            computation["acceleration"]
        )
    )

    # --- MATERIAL
    parameters = {"YoungModulus": 28.85e6, "PoissonRatio": 0.499}
    stabilization_parameter = 1.0 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
    mat = Material(
        nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
        library_path="behaviours/bhv_small_strain_perfect_plasticity/src/libBehaviour.so",
        library_name="SmallStrainPerfectPlasticity",
        hypothesis=mgis_bv.Hypothesis.AXISYMMETRICAL,
        stabilization_parameter=stabilization_parameter,
        lagrange_parameter=stabilization_parameter,
        field=displacement,
        integration_type=jacobian_rule[computation["jacobian"]],
        parameters=None,
    )

    # --- SOLVE
    solve(
        p,
        mat,
        verbose=False,
        debug_mode=DebugMode.NONE,
        accelerate=acceleration_rule[computation["acceleration"]],
        num_local_iterations=algorithm_rule[computation["algorithm"]]
    )

def solve_notched_rod(computation):
    # --- VALUES
    time_steps = np.linspace(0.0, 0.0008, 100)
    iterations = 200

    # --- LOAD
    def volumetric_load(time: float, position: ndarray):
        return 0

    loads = [Load(volumetric_load, 0), Load(volumetric_load, 1)]

    # --- BC
    def pull(time: float, position: ndarray) -> float:
        return time

    def clamped(time: float, position: ndarray) -> float:
        return 0.0

    boundary_conditions = [
        BoundaryCondition("TOP", pull, BoundaryType.DISPLACEMENT, 1),
        BoundaryCondition("BOTTOM", clamped, BoundaryType.DISPLACEMENT, 1),
        BoundaryCondition("LEFT", clamped, BoundaryType.DISPLACEMENT, 0),
    ]

    # --- MESH
    mesh_file_path = "meshes/notched_rod/notched_rod_{}.msh".format(computation["mesh"])

    # --- FIELD
    displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC)

    # --- FINITE ELEMENT
    finite_element = FiniteElement(
        element_type=ElementType.HHO_HIGH,
        polynomial_order=computation["order"],
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
        tolerance=1.0e-6,
        res_folder_path=get_current_res_folder_path() + "_{}_notched_rod__{}__hho{}__{}__{}__{}".format(
            batch,
            computation["mesh"],
            computation["order"],
            computation["algorithm"],
            computation["jacobian"],
            computation["acceleration"]
        )
    )

    # --- MATERIAL
    parameters = {"YoungModulus": 206.9e9, "PoissonRatio": 0.29}
    stabilization_parameter = parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
    mat = Material(
        nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
        library_path="behaviours/bhv_large_strain_voce_plasticity/src/libBehaviour.so",
        library_name="LargeStrainVocePlasticity",
        hypothesis=mgis_bv.Hypothesis.AXISYMMETRICAL,
        stabilization_parameter=stabilization_parameter,
        lagrange_parameter=parameters["YoungModulus"],
        field=displacement,
        integration_type=jacobian_rule[computation["jacobian"]],
        parameters=None,
    )

    # --- SOLVE
    solve(
        p,
        mat,
        verbose=False,
        debug_mode=DebugMode.NONE,
        accelerate=acceleration_rule[computation["acceleration"]],
        num_local_iterations=algorithm_rule[computation["algorithm"]]
    )

computation_parameters = [
    # --- ORDER 1
    # CONSISTENT JACOBIAN
    {"mesh" : "coarse", "order": 1, "algorithm" : "Explicit", "jacobian" : "Consistent", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 1, "algorithm" : "Implicit", "jacobian" : "Consistent", "acceleration": "NonAccelerated"},
    # ELASTIC JACOBIAN
    {"mesh" : "coarse", "order": 1, "algorithm" : "Explicit", "jacobian" : "Elastic", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 1, "algorithm" : "Explicit", "jacobian" : "Elastic", "acceleration": "Accelerated"},
    {"mesh" : "coarse", "order": 1, "algorithm" : "Implicit", "jacobian" : "Elastic", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 1, "algorithm" : "Implicit", "jacobian" : "Elastic", "acceleration": "Accelerated"},
    # --- ORDER 2
    # CONSISTENT JACOBIAN
    {"mesh" : "coarse", "order": 2, "algorithm" : "Explicit", "jacobian" : "Consistent", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 2, "algorithm" : "Implicit", "jacobian" : "Consistent", "acceleration": "NonAccelerated"},
    # ELASTIC JACOBIAN
    {"mesh" : "coarse", "order": 2, "algorithm" : "Explicit", "jacobian" : "Elastic", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 2, "algorithm" : "Explicit", "jacobian" : "Elastic", "acceleration": "Accelerated"},
    {"mesh" : "coarse", "order": 2, "algorithm" : "Implicit", "jacobian" : "Elastic", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 2, "algorithm" : "Implicit", "jacobian" : "Elastic", "acceleration": "Accelerated"},
    # --- ORDER 3
    # CONSISTENT JACOBIAN
    {"mesh" : "coarse", "order": 3, "algorithm" : "Explicit", "jacobian" : "Consistent", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 3, "algorithm" : "Implicit", "jacobian" : "Consistent", "acceleration": "NonAccelerated"},
    # ELASTIC JACOBIAN
    {"mesh" : "coarse", "order": 3, "algorithm" : "Explicit", "jacobian" : "Elastic", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 3, "algorithm" : "Explicit", "jacobian" : "Elastic", "acceleration": "Accelerated"},
    {"mesh" : "coarse", "order": 3, "algorithm" : "Implicit", "jacobian" : "Elastic", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 3, "algorithm" : "Implicit", "jacobian" : "Elastic", "acceleration": "Accelerated"},
]

for computation in computation_parameters:
    solve_swelling_sphere(computation)

for computation in computation_parameters:
    solve_notched_rod(computation)