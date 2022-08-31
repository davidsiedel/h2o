from unittest import TestCase

import numpy as np

from h2o.problem.problem import Problem
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.material import Material
from h2o.problem.load import Load
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
from h2o.h2o import *
from mgis import behaviour as mgis_bv

from h2o.problem.resolution.static_condensation import solve_newton_static_condensation
from h2o.problem.resolution.static_condensation import solve_newton_static_condensation
from h2o.problem.resolution.exact import solve_newton_exact
from h2o.problem.resolution.local_equilibrium import solve_newton_local_equilibrium


class TestMecha(TestCase):
    def test_cook_finite_strain_voce_isotropic_hardening(self):
        # --- VALUES
        time_steps = np.linspace(0.0, 7.0e-3, 50)
        time_steps = np.linspace(0.0, 14.e-3, 150)
        P_min = 0.0
        P_max = 5.e6 / (16.e-3)
        # P_max = 5.e3 / (16.e-3)
        # P_max = 3.e8
        # P_min = 0.01
        # P_max = 1. / 16.
        # time_steps = np.linspace(P_min, P_max, 20)[:-3]
        time_steps = np.linspace(P_min, P_max, 15)
        # time_steps = np.linspace(0.0, 0.004, 25)
        # time_steps = list(time_steps) + [P_max]
        time_steps = list(time_steps)
        print(time_steps)
        iterations = 10

        # --- LOAD
        def volumetric_load(time: float, position: ndarray):
            return 0.0

        loads = [Load(volumetric_load, 0), Load(volumetric_load, 1)]

        # --- BC
        def pull(time: float, position: ndarray) -> float:
            return time

        def fixed(time: float, position: ndarray) -> float:
            return 0.0

        boundary_conditions = [
            BoundaryCondition("RIGHT", pull, BoundaryType.PRESSURE, 1),
            # BoundaryCondition("RIGHT", pull, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
        ]

        # --- MESH
        # mesh_file_path = "meshes/cook_5.geof"
        # mesh_file_path = "meshes/cook_30.geof"
        # mesh_file_path = "meshes/cook_quadrangles_0.msh"
        # mesh_file_path = "meshes/cook_20_quadrangles_structured.msh"
        mesh_file_path = "meshes/cook_01_quadrangles_structured.msh"
        # mesh_file_path = "meshes/cook_02_quadrangles_structured.msh"
        # mesh_file_path = "meshes/cook_03_quadrangles_structured.msh"
        # mesh_file_path = "meshes/cook_10_triangles_structured.msh"
        # mesh_file_path = "meshes/cook_10_quadrangles_structured.msh"
        # mesh_file_path = "meshes/cook_16_triangles_structured.msh"
        # ----
        # mesh_file_path = "meshes/cook_compare_m.geof"
        # mesh_file_path = "meshes/cook_32_triangles_structured.msh"
        # mesh_file_path = "meshes/cook_32_quadrangles_structured.msh"
        # mesh_file_path = "meshes2/cook_soda_45.msh"
        # mesh_file_path = "meshes2/cook_45_quadrangles_structured2.msh"
        # mesh_file_path = "meshes2/cook_5.msh"

        # --- FIELD
        displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN)

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
            tolerance=1.0e-6,
            res_folder_path=get_current_res_folder_path() + "_HDG_K2L2"
        )

        # --- MATERIAL
        parameters = {"YoungModulus": 206.9e9, "PoissonRatio": 0.29}
        stabilization_parameter = 0.00005 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"]) #GOOD
        # stabilization_parameter = 0.05 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"]) #GOOD
        stabilization_parameter = 1.0 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        print("STAB : {}".format(stabilization_parameter))
        mat = Material(
            nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
            library_path="behaviour/src/libBehaviour.so",
            library_name="Voce",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=stabilization_parameter,
            lagrange_parameter=parameters["YoungModulus"],
            field=displacement,
            parameters=None,
        )

        # --- SOLVE
        # solve_newton_static_condensation(p, mat, verbose=False, debug_mode=DebugMode.NONE)
        # solve_newton_local(p, mat, verbose=False, debug_mode=DebugMode.NONE)
        solve_newton_static_condensation(p, mat, verbose=False, debug_mode=DebugMode.NONE)
        # solve_newton_local_equilibrium(p, mat, verbose=False, debug_mode=DebugMode.NONE)