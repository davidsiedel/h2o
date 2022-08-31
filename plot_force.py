import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os import path, walk
from typing import List

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 12}
#
# matplotlib.rc('font', **font)
plt.rcParams.update({'font.size': 14})

res_folder = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_geof"


def __get_reaction_curve(res_folder: str, num_time_steps: int):
    _, _, filenames = next(walk(res_folder))
    forces = []
    for time_step_index in range(num_time_steps):
        for filename in filenames:
            if "{}".format(time_step_index).zfill(6) in filename and "qdp" in filename:
                hho_file_path = path.join(res_folder, filename)
                with open(hho_file_path, "r") as hho_res_file:
                    index = 10459
                    c_hho = hho_res_file.readlines()
                    line = c_hho[index]
                    x_coordinates = float(line.split(",")[0])
                    y_coordinates = float(line.split(",")[1])
                    x_disp = float(line.split(",")[2])
                    sig_11 = float(line.split(",")[10])
                    force = sig_11 * ((0.0054 + x_disp))
                    print("done")
                    forces.append(force)
    return forces
    # plt.plot(time_steps, forces, label="python HHO")
    # plt.plot(cast_times, cast_forces, label="Cast3M", linestyle='--')
    # plt.legend()
    # plt.xlabel("displacement [m]")
    # plt.ylabel("reaction force [N]")
    # plt.grid()
    # plt.show()

catsem_res_path = "/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/SSNA303_FU.csv"
catsem_res_tri3_path = "/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/SSNA303_FU.csv"
castem_time = []
castem_force = []
with open(catsem_res_path, "r") as castem_res:
    c = castem_res.readlines()
    for i, line in enumerate(c):
        if i > 0 and i < 274:
            time = float(line.split("  ")[1])
            force = float(line.split("  ")[2])
            castem_time.append(time)
            castem_force.append(force)
castem_time2 = []
castem_force2 = []
with open(catsem_res_tri3_path, "r") as castem_res:
    c = castem_res.readlines()
    for i, line in enumerate(c):
        if i > 0 and i < 274:
            time = float(line.split("  ")[1])
            force = float(line.split("  ")[2])
            castem_time2.append(time)
            castem_force2.append(force)
hho_res_path = "/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_geoflocal/output_LRD.csv"
# hho_res_path = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_msh/output_BOTTOM.csv"
hho_time = []
hho_force = []
with open(hho_res_path, "r") as hho_res:
    c = hho_res.readlines()
    for i, line in enumerate(c):
        if i > 0 and i < 78:
            time = float(line.split(",")[0])
            force = np.abs(float(line.split(",")[1]))
            hho_time.append(time)
            hho_force.append(force)
plt.xlabel("displacment [m]")
plt.ylabel("force [MN]")
plt.grid()
# plt.plot(castem_time2, castem_force2, label="CAST3M TRI3", c="green", linestyle="--", linewidth=2)
plt.plot(np.array(castem_time), np.array(castem_force)/1.e6, label="CAST3M TRI6", c="red", linestyle="--", linewidth=3)
plt.scatter(np.array(hho_time), np.array(hho_force)/1.e6, label="HHO", c="blue", s=50, marker='o')
# forces = __get_reaction_curve(res_folder, len(hho_time))
# plt.plot(hho_time, forces, label="OTHER", c="green")
# plt.scatter(castem_time, castem_force, label="CAST3M", c="red")
# plt.scatter(hho_time, hho_force, label="HHO", c="blue")
plt.legend()
plt.show()