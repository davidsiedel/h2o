from unittest import TestCase

from h2o.problem.problem import Problem
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.material import Material
from h2o.problem.load import Load
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
from h2o.h2o import *
from mgis import behaviour as mgis_bv

from h2o.problem.resolution.condensation import solve_newton_2
from h2o.problem.resolution.exact import solve_newton_exact


class TestMecha(TestCase):
    def test_square_finite_strain_linear_isotropic_hardening(self):
        # --- VALUES
        spacing = 3
        time_steps_1 = np.linspace(0.0, 7.0e-3, spacing)
        time_steps_2 = np.linspace(7.0e-3, -1.0e-2, spacing)
        time_steps_3 = np.linspace(-1.0e-2, 2.0e-2, spacing)
        time_steps_4 = np.linspace(2.0e-2, -3.0e-2, spacing)
        time_steps_5 = np.linspace(-3.0e-2, 4.0e-2, spacing)
        time_steps = []
        for ts in [time_steps_1, time_steps_2[1:], time_steps_3[1:], time_steps_4[1:], time_steps_5[1:]]:
            # time_steps += list(np.sqrt(2.)*ts)
            time_steps += list(ts)
        time_steps = np.array(time_steps)
        iterations = 100

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
            BoundaryCondition("RIGHT", pull, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 1),
        ]

        # --- MESH
        mesh_file_path = (
            # "meshes/triang_r.geof"
            # "meshes/triang_2.geof"
            # "meshes/square_1.geof"
            # "meshes/pentag_1.geof"
            # "meshes/triangles_0.msh"
            # "meshes/quadrangles_3.msh"
            "meshes/quadrangles_2.msh"
            # "meshes/triang_3.geof"
        )

        # --- FIELD
        displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN)

        # --- FINITE ELEMENT
        finite_element = FiniteElement(
            element_type=ElementType.HDG_EQUAL,
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
            tolerance=1.0e-4,
            res_folder_path=get_current_res_folder_path()
        )

        # --- MATERIAL
        parameters = {"YoungModulus": 70.0e9, "PoissonRatio": 0.34, "HardeningSlope": 10.0e9, "YieldStress": 300.0e6}
        stabilization_parameter = parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        mat = Material(
            nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
            library_path="behaviour/src/libBehaviour.so",
            library_name="IsotropicLinearHardeningPlasticity",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=stabilization_parameter,
            lagrange_parameter=parameters["YoungModulus"],
            field=displacement,
            parameters=None,
            # finite_strains=False
        )

        # --- SOLVE
        solve_newton_2(p, mat, verbose=False)
        # solve_newton_exact(p, mat, verbose=False)

        # --- POST PROCESSING
        from pp.plot_data import plot_data

        mtest_file_path = "mtest/finite_strain_isotropic_linear_hardening.res"
        hho_res_dir_path = "res"
        number_of_time_steps = len(time_steps)
        m_x_inedx = 1
        m_y_index = 6
        d_x_inedx = 4
        d_y_inedx = 9
        plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        m_x_inedx = 1
        m_y_index = 7
        d_x_inedx = 4
        d_y_inedx = 10
        plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        m_x_inedx = 1
        m_y_index = 8
        d_x_inedx = 4
        d_y_inedx = 11
        plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        m_x_inedx = 1
        m_y_index = 9
        d_x_inedx = 4
        d_y_inedx = 12
        plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
