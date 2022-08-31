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
from h2o.problem.resolution.exact import solve_newton_exact


class TestMecha(TestCase):
    def test_rod_finite_strain_voce_isotropic_hardening(self):
        time_steps = 1.e-3 * np.array([
            0.0,
            0.232497066547,
            0.436497737051,
            0.539939654691,
            0.639939654691,
            0.843381572331,
            1.03782756887,
            1.2494831536,
            1.45236631838,
            1.64625356205,
            1.84868972453,
            2.05123763759,
            2.24467787897,
            2.44689054031,
            2.64921495223,
            2.85120411242,
            3.0617421914,
            3.24596301056,
            # 3.45616583785,
            # 3.66625691457,
            # 3.85869140079,
            # 4.04212996592,
            # 4.24311337096,
            # 4.44398502542,
            # 4.64474492932,
            # 4.86304967313,
            # 5.0282728949
        ])
        # --- VALUES
        u_min = 0.0
        # u_max = 0.0003
        u_max = 0.005
        ts = np.linspace(u_min, u_max, 100)
        # ts = np.linspace(0.0, 0.00324596301056, 50)
        # ts = np.linspace(0.0, 0.00404212996592, 100)
        ts = np.linspace(0.0, 4.04212996592, 100)
        time_steps = list(ts)
        # ts0 = np.linspace(0.0, 0.0006, 100)
        # ts1 = np.linspace(0.0006, 0.0009, 200)
        # # time_steps = list(time_steps) + [u_max]
        # # time_steps = list(ts)
        # time_steps = list(ts0) + list(ts1)
        print(time_steps)
        iterations = 10

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
            BoundaryCondition("TOP", pull, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
        ]

        # --- MESH
        mesh_file_path = "../test_notched_specimen_finite_strain_isotropic_voce_hardening/meshes/ssna303_strcut_qua_1.msh"
        # mesh_file_path = "meshes/ssna303_strcut_qua_0.msh"
        mesh_file_path = "meshes/rectangle_rod3.msh"
        # mesh_file_path = "meshes/defected_rod_quadrangles_0.msh"
        mesh_file_path = "meshes/defected_rod_quadrangles_1.msh"
        # mesh_file_path = "meshes/rectangle_rod4.msh"
        # mesh_file_path = "meshes/rectangle_rod_tri2.msh"
        # mesh_file_path = "meshes/rectangle_rod4_straight.msh"

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
            tolerance=1.0e-6,
            res_folder_path=get_current_res_folder_path()
        )

        # --- MATERIAL
        # parameters = {"YoungModulus": 206.9e9, "PoissonRatio": 0.29}
        parameters = {"YoungModulus": 206.9e3, "PoissonRatio": 0.29}
        stabilization_parameter = 1.0 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        print("STAB : {}".format(stabilization_parameter))
        mat = Material(
            nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
            library_path="behaviour/src/libBehaviour.so",
            library_name="Voce",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=stabilization_parameter,
            # lagrange_parameter=parameters["YoungModulus"],
            lagrange_parameter=stabilization_parameter,
            field=displacement,
            parameters=None,
        )

        # --- SOLVE
        solve_newton_static_condensation(p, mat, verbose=False, debug_mode=DebugMode.NONE)

        from os import walk, path
        import matplotlib.pyplot as plt

        plt.plot(boundary_conditions[0].time_values, boundary_conditions[0].force_values)
        plt.grid()
        plt.show()

        res_x_nic = 1.e-3 * np.array([
            0.00877241995865,
            0.232497066547,
            0.436497737051,
            0.639939654691,
            0.843381572331,
            1.03782756887,
            1.2494831536,
            1.45236631838,
            1.64625356205,
            1.84868972453,
            2.05123763759,
            2.24467787897,
            2.44689054031,
            2.64921495223,
            2.85120411242,
            3.0617421914,
            3.24596301056,
            3.45616583785,
            3.66625691457,
            3.85869140079,
            4.04212996592,
            4.24311337096,
            4.44398502542,
            4.64474492932,
            4.86304967313,
            5.0282728949]
        )

        res_y_nic = 1.e6 * np.array([
            0.0127395652903,
            3.49187014583,
            3.74794658323,
            3.94032519417,
            4.13270380511,
            4.29954741018,
            4.42828406996,
            4.55696485444,
            4.66011063307,
            4.73783315639,
            4.82829524501,
            4.88048276247,
            4.93272615522,
            4.99770911326,
            5.02447337543,
            5.0258143823,
            5.02698776331,
            4.99011007431,
            4.94049282003,
            4.87802424987,
            4.79002067386,
            4.70212884841,
            4.60149745767,
            4.48812650165,
            4.37486729619,
            4.21031457786]
        )
        res_folder = get_current_res_folder_path()
        def __plot_reaction_curve():
            _, _, filenames = next(walk(res_folder))
            forces = []
            disps = []
            force_item = 0.
            disp_item = 0.
            sig_item = 0.
            section_item = 0.
            for time_step_index in range(len(time_steps)+1):
                for filename in filenames:
                    if "{}".format(time_step_index).zfill(6) in filename and "qdp" in filename:
                        hho_file_path = path.join(res_folder, filename)
                        with open(hho_file_path, "r") as hho_res_file:
                            c_hho = hho_res_file.readlines()
                            index = 2
                            index = 8
                            # index = 4
                            # index = 97
                            index_0 = 3
                            y_coordinates_0 = float(c_hho[index_0].split(",")[1])
                            sig11_0 = float(c_hho[index_0].split(",")[10])
                            index_1 = 4
                            y_coordinates_1 = float(c_hho[index_1].split(",")[1])
                            sig11_1 = float(c_hho[index_1].split(",")[10])
                            sig_item = sig11_0 + np.abs((sig11_1 - sig11_0)/(y_coordinates_1-y_coordinates_0)) * (0.026667 - y_coordinates_0)
                            line = c_hho[index]
                            x_coordinates = float(line.split(",")[0])
                            y_coordinates = float(line.split(",")[1])
                            x_disp = float(line.split(",")[2])
                            y_disp = float(line.split(",")[3])
                            sig_11 = float(line.split(",")[10])
                            # sig_11 = float(line.split(",")[11])
                            # force = sig_11 * ((0.0054 + x_disp))
                            # print(x_coordinates)
                            force = sig_11 * ((0.006413 + x_disp))
                            # force = sig_11 * ((x_coordinates + x_disp))
                            force_item = sig_11
                            # forces.append(force)
                            # disps.append(np.abs(y_disp))
                    if "{}".format(time_step_index).zfill(6) in filename and "vtx" in filename:
                        hho_file_path = path.join(res_folder, filename)
                        print(filename)
                        print(time_step_index)
                        with open(hho_file_path, "r") as hho_res_file:
                            index = 4
                            c_hho = hho_res_file.readlines()
                            line = c_hho[index]
                            x_coordinates = float(line.split(",")[0])
                            y_coordinates = float(line.split(",")[1])
                            x_disp = float(line.split(",")[2])
                            y_disp = float(line.split(",")[3])
                            section_item = x_coordinates + x_disp
                            disp_item = y_disp
                            print(disp_item)
                # forces.append(force_item * section_item)
                forces.append(sig_item * section_item)
                disps.append(np.abs(disp_item))

            # cast_forces = []
            # cast_times = []
            # with open("castem/SSNA303_FU.csv", "r") as castfile:
            #     cast_c = castfile.readlines()
            #     for line in cast_c:
            #         cast_force = float(line.split(",")[2])
            #         cast_time = float(line.split(",")[1])
            #         cast_forces.append(cast_force)
            #         cast_times.append(cast_time)
            # plt.plot(time_steps, forces, label="python HHO")
            # plt.plot(np.linspace(u_min, u_max, len(forces)), forces, label="python HHO")
            # print(disps)
            # plt.plot(disps, forces, label="python HHO")
            plt.plot(boundary_conditions[1].time_values, boundary_conditions[1].force_values, label="python HHO")
            plt.plot(res_x_nic, res_y_nic, label="Pignet, 2019")
            # plt.plot(cast_times, cast_forces, label="Cast3M", linestyle='--')
            plt.legend()
            plt.xlabel("displacement [m]")
            # plt.xlim(0., 0.005)
            # plt.ylim(0., 6.e6)
            plt.ylabel("reaction force [N]")
            plt.grid()
            plt.show()

        __plot_reaction_curve()

        # solve_newton_exact(p, mat, verbose=False, debug_mode=DebugMode.NONE)

        # from pp.plot_ssna import plot_det_f
        #
        # # plot_det_f(46, "res")
        #
        # res_folder = "res"
        # # res_folder = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_cook_finite_strain_isotropic_voce_hardening/res_cook_20_ord1_quad/res"
        # from os import walk, path
        # import matplotlib.pyplot as plt
        # from matplotlib.colors import LinearSegmentedColormap
        #
        # def __plot(column: int, time_step_index: int):
        #
        #     _, _, filenames = next(walk(res_folder))
        #     # for time_step_index in range(1, len(time_steps)):
        #     # for time_step_index in range(30, len(time_steps)):
        #     for filename in filenames:
        #         if "{}".format(time_step_index).zfill(6) in filename and "qdp" in filename:
        #             hho_file_path = path.join(res_folder, filename)
        #             with open(hho_file_path, "r") as hho_res_file:
        #                 fig, ax0d = plt.subplots(nrows=1, ncols=1)
        #                 c_hho = hho_res_file.readlines()
        #                 field_label = c_hho[0].split(",")[column]
        #                 number_of_points = len(c_hho) - 1
        #                 # for _iloc in range(len(c_hho)):
        #                 #     line = c_hho[_iloc]
        #                 #     x_coordinates = float(line.split(",")[0])
        #                 #     y_coordinates = float(line.split(",")[1])
        #                 #     if (x_coordinates - 0.0) ** 2 + (y_coordinates)
        #                 eucli_d = displacement.euclidean_dimension
        #                 points = np.zeros((eucli_d, number_of_points), dtype=real)
        #                 field_vals = np.zeros((number_of_points,), dtype=real)
        #                 field_min_val = np.inf
        #                 field_max_val = -np.inf
        #                 for l_count, line in enumerate(c_hho[1:]):
        #                     x_coordinates = float(line.split(",")[0])
        #                     y_coordinates = float(line.split(",")[1])
        #                     field_value = float(line.split(",")[column])
        #                     points[0, l_count] += x_coordinates
        #                     points[1, l_count] += y_coordinates
        #                     field_vals[l_count] += field_value
        #                     # if field_vals[l_count]
        #                 x, y = points
        #                 colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
        #                 # perso = LinearSegmentedColormap.from_list("perso", colors, N=1000)
        #                 perso = LinearSegmentedColormap.from_list("perso", colors, N=20)
        #                 vmin = min(field_vals[:])
        #                 vmax = max(field_vals[:])
        #                 # vmin = 300.e6
        #                 # vmax = 400.e6
        #                 # vmin = 8.e8/3.
        #                 # vmax = 12.e8/3.
        #                 # levels = np.linspace(vmin, vmax, 50, endpoint=True)
        #                 levels = np.linspace(vmin, vmax, 20, endpoint=True)
        #                 ticks = np.linspace(vmin, vmax, 10, endpoint=True)
        #                 datad = ax0d.tricontourf(x, y, field_vals[:], cmap=perso, levels=levels)
        #                 ax0d.get_xaxis().set_visible(False)
        #                 ax0d.get_yaxis().set_visible(False)
        #                 ax0d.set_xlabel("map of the domain $\Omega$")
        #                 cbar = fig.colorbar(datad, ax=ax0d, ticks=ticks)
        #                 cbar.set_label("{} : {}".format(field_label, time_step_index), rotation=270, labelpad=15.0)
        #                 # plt.savefig("/home/dsiedel/Projects/pythhon/plots/{}.png".format(time_step))
        #                 plt.show()
        #
        # for tsindex in [1, 50, 100, 192]:
        #     # __plot(15, tsindex)
        #     pass
        # # __plot(15, 19)
        # __plot(15, 34)
        # # __plot(15, 37)
        # # __plot(3)
        #
        # # --- POST PROCESSING
        # # from pp.plot_data import plot_data
        # # mtest_file_path = "mtest/finite_strain_isotropic_linear_hardening.res"
        # # hho_res_dir_path = "../../../res"
        # # number_of_time_steps = len(time_steps)
        # # m_x_inedx = 1
        # # m_y_index = 6
        # # d_x_inedx = 4
        # # d_y_inedx = 9
        # # plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        # # m_x_inedx = 1
        # # m_y_index = 7
        # # d_x_inedx = 4
        # # d_y_inedx = 10
        # # plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        # # m_x_inedx = 1
        # # m_y_index = 8
        # # d_x_inedx = 4
        # # d_y_inedx = 11
        # # plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        # # m_x_inedx = 1
        # # m_y_index = 9
        # # d_x_inedx = 4
        # # d_y_inedx = 12
        # # plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
