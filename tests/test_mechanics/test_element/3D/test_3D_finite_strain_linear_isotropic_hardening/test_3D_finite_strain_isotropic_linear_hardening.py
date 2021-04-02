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
    def test_cube_finite_strain_isotropic_linear_hardening(self):
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

        loads = [Load(volumetric_load, 0), Load(volumetric_load, 1), Load(volumetric_load, 2)]

        # --- BC
        def pull(time: float, position: ndarray) -> float:
            return time

        def fixed(time: float, position: ndarray) -> float:
            return 0.0

        boundary_conditions = [
            BoundaryCondition("RIGHT", pull, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("BACK", fixed, BoundaryType.DISPLACEMENT, 2),
        ]

        # --- MESH
        # mesh_file_path = "meshes/hexahe_1.geof"
        # mesh_file_path = "meshes/polyhe_1.geof"
        # mesh_file_path = "meshes/tetrahedra_0.msh"
        # mesh_file_path = "meshes/tetrahedra_3.msh"
        mesh_file_path = "meshes/hexahedra_0.msh"
        # mesh_file_path = "meshes/tetrahedra_1.msh"

        # --- FIELD
        displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_LARGE_STRAIN)

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
        parameters = {"YoungModulus": 70.0e9, "PoissonRatio": 0.34}
        stabilization_parameter = parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        # stabilization_parameter = 0.001 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        mat = Material(
            nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
            library_path="behaviour/src/libBehaviour.so",
            library_name="IsotropicLinearHardeningPlasticity",
            hypothesis=mgis_bv.Hypothesis.TRIDIMENSIONAL,
            stabilization_parameter=stabilization_parameter,
            lagrange_parameter=parameters["YoungModulus"],
            field=displacement,
            parameters=None,
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
        m_y_index = 10
        d_x_inedx = 6
        d_y_inedx = 15
        plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        m_x_inedx = 1
        m_y_index = 11
        d_x_inedx = 6
        d_y_inedx = 16
        plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        m_x_inedx = 1
        m_y_index = 12
        d_x_inedx = 6
        d_y_inedx = 17
        plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        m_x_inedx = 1
        m_y_index = 13
        d_x_inedx = 6
        d_y_inedx = 18
        plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        m_x_inedx = 1
        m_y_index = 14
        d_x_inedx = 6
        d_y_inedx = 20
        plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        m_x_inedx = 1
        m_y_index = 15
        d_x_inedx = 6
        d_y_inedx = 22
        plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
