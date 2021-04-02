import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os import walk, path

from matplotlib.colors import LinearSegmentedColormap

from h2o.h2o import *


def plot_det_f(time_step_index: int, res_folder: str):
    _, _, filenames = next(walk(res_folder))
    for filename in filenames:
        if "{}".format(time_step_index).zfill(6) in filename and "qdp" in filename:
            hho_file_path = path.join(res_folder, filename)
            with open(hho_file_path, "r") as hho_res_file:
                fig, ax0d = plt.subplots(nrows=1, ncols=1, figsize=(11, 15))
                matplotlib.rcParams.update({'font.size': 22})
                c_hho = hho_res_file.readlines()
                field_label = "DET_F"
                number_of_points = len(c_hho) - 1
                F = np.zeros((number_of_points, 3, 3), dtype=real)
                eucli_d = 2
                points = np.zeros((eucli_d, number_of_points), dtype=real)
                field_vals = np.zeros((number_of_points,), dtype=real)
                for l_count, line in enumerate(c_hho[1:]):
                    x_coordinates = float(line.split(",")[0])
                    y_coordinates = float(line.split(",")[1])
                    F_00 = float(line.split(",")[4])
                    F_11 = float(line.split(",")[5])
                    F_22 = float(line.split(",")[6])
                    F_01 = float(line.split(",")[7])
                    F_10 = float(line.split(",")[8])
                    F[l_count, 0, 0] = F_00
                    F[l_count, 1, 1] = F_11
                    F[l_count, 2, 2] = F_22
                    F[l_count, 0, 1] = F_01
                    F[l_count, 1, 0] = F_10
                    det_F = np.linalg.det(F[l_count])
                    points[0, l_count] += x_coordinates
                    points[1, l_count] += y_coordinates
                    field_vals[l_count] += det_F
                x, y = points
                colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
                perso = LinearSegmentedColormap.from_list("perso", colors, N=1000)
                vmin = min(field_vals[:])
                vmax = max(field_vals[:])
                vmin = 9.84e-1
                vmax = 1.08
                levels = np.linspace(vmin, vmax, 1000, endpoint=True)
                ticks = np.linspace(vmin, vmax, 10, endpoint=True)
                datad = ax0d.tricontourf(x, y, field_vals[:], cmap=perso, levels=levels)
                ax0d.get_xaxis().set_visible(False)
                ax0d.get_yaxis().set_visible(False)
                # ax0d.set_xlabel("map of the domain $\Omega$")
                cbar = fig.colorbar(datad, ax=ax0d, ticks=ticks)
                cbar.set_label("{}".format(field_label), rotation=270, labelpad=15.0)
                # plt.savefig("/home/dsiedel/Projects/pythhon/plots/{}.png".format(time_step))
                plt.show()