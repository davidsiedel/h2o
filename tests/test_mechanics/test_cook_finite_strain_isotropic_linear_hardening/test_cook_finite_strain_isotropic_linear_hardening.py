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
    def test_cook_finite_strain_linear_isotropic_hardening(self):
        # --- VALUES
        time_steps = np.linspace(0.0, 6.0e-3, 50)
        P_min = 0.01
        P_max = 70.e9 / 16.
        # P_min = 0.01
        # P_max = 1. / 16.
        time_steps = np.linspace(P_min, P_max, 20)[:-3]
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
            BoundaryCondition("RIGHT", pull, BoundaryType.PRESSURE, 1),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
        ]

        # --- MESH
        # mesh_file_path = "meshes/cook_quadrangles_1.msh"
        # mesh_file_path = "meshes/cook_quadrangles_0.msh"
        mesh_file_path = "meshes/cook_triangles_0.msh"

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
        # stabilization_parameter = 1000. * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        stabilization_parameter = 0.001 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        stabilization_parameter = 1.0 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        mat = Material(
            nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
            library_path="behaviour/src/libBehaviour.so",
            library_name="FiniteStrainIsotropicLinearHardeningPlasticity",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=stabilization_parameter,
            lagrange_parameter=parameters["YoungModulus"],
            field=displacement,
            parameters=None,
        )

        # --- SOLVE
        solve_newton_2(p, mat, verbose=False, debug_mode=DebugMode.NONE)
        # solve_newton_exact(p, mat, verbose=False, debug_mode=DebugMode.NONE)

        from pp.plot_ssna import plot_det_f

        # plot_det_f(46, "res")

        res_folder = "res"
        from os import walk, path
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        def __plot(column: int):

            _, _, filenames = next(walk(res_folder))
            for time_step_index in range(1, len(time_steps)):
                for filename in filenames:
                    if "{}".format(time_step_index).zfill(6) in filename and "qdp" in filename:
                        hho_file_path = path.join(res_folder, filename)
                        with open(hho_file_path, "r") as hho_res_file:
                            fig, ax0d = plt.subplots(nrows=1, ncols=1)
                            c_hho = hho_res_file.readlines()
                            field_label = c_hho[0].split(",")[column]
                            number_of_points = len(c_hho) - 1
                            # for _iloc in range(len(c_hho)):
                            #     line = c_hho[_iloc]
                            #     x_coordinates = float(line.split(",")[0])
                            #     y_coordinates = float(line.split(",")[1])
                            #     if (x_coordinates - 0.0) ** 2 + (y_coordinates)
                            eucli_d = displacement.euclidean_dimension
                            points = np.zeros((eucli_d, number_of_points), dtype=real)
                            field_vals = np.zeros((number_of_points,), dtype=real)
                            for l_count, line in enumerate(c_hho[1:]):
                                x_coordinates = float(line.split(",")[0])
                                y_coordinates = float(line.split(",")[1])
                                field_value = float(line.split(",")[column])
                                points[0, l_count] += x_coordinates
                                points[1, l_count] += y_coordinates
                                field_vals[l_count] += field_value
                            x, y = points
                            colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
                            perso = LinearSegmentedColormap.from_list("perso", colors, N=1000)
                            vmin = min(field_vals[:])
                            vmax = max(field_vals[:])
                            # vmin = -3.656e9
                            # vmax = 1.214e9
                            levels = np.linspace(vmin, vmax, 50, endpoint=True)
                            ticks = np.linspace(vmin, vmax, 10, endpoint=True)
                            datad = ax0d.tricontourf(x, y, field_vals[:], cmap=perso, levels=levels)
                            ax0d.get_xaxis().set_visible(False)
                            ax0d.get_yaxis().set_visible(False)
                            ax0d.set_xlabel("map of the domain $\Omega$")
                            cbar = fig.colorbar(datad, ax=ax0d, ticks=ticks)
                            cbar.set_label("{}".format(field_label), rotation=270, labelpad=15.0)
                            # plt.savefig("/home/dsiedel/Projects/pythhon/plots/{}.png".format(time_step))
                            plt.show()

        __plot(15)

        # --- POST PROCESSING
        # from pp.plot_data import plot_data
        # mtest_file_path = "mtest/finite_strain_isotropic_linear_hardening.res"
        # hho_res_dir_path = "../../../res"
        # number_of_time_steps = len(time_steps)
        # m_x_inedx = 1
        # m_y_index = 6
        # d_x_inedx = 4
        # d_y_inedx = 9
        # plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        # m_x_inedx = 1
        # m_y_index = 7
        # d_x_inedx = 4
        # d_y_inedx = 10
        # plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        # m_x_inedx = 1
        # m_y_index = 8
        # d_x_inedx = 4
        # d_y_inedx = 11
        # plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        # m_x_inedx = 1
        # m_y_index = 9
        # d_x_inedx = 4
        # d_y_inedx = 12
        # plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
