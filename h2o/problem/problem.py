from h2o.mesh.mesh import Mesh
from h2o.fem.element.element import Element
from h2o.geometry.shape import Shape
from h2o.fem.element.finite_element import FiniteElement
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.load import Load
from h2o.problem.material import Material
from h2o.field.field import Field
from h2o.h2o import *

# from scipy.sparse.linalg import spsolve
# from scipy.sparse import csr_matrix
#
# from mgis import behaviour as mgis_bv


def clean_res_dir():
    """

    """
    res_folder = os.path.join(get_project_path(), "res")
    print(res_folder)
    for filename in os.listdir(res_folder):
        file_path = os.path.join(res_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


class Problem:
    finite_element: FiniteElement
    field: Field
    mesh: Mesh
    boundary_conditions: List[BoundaryCondition]
    loads: List[Load]
    time_steps: ndarray
    number_of_iterations: int
    tolerance: float
    elements: List[Element]

    def __init__(
        self,
        mesh_file_path: str,
        field: Field,
        time_steps: ndarray,
        iterations: int,
        finite_element: FiniteElement,
        boundary_conditions: List[BoundaryCondition],
        loads: List[Load] = None,
        quadrature_type: QuadratureType = QuadratureType.GAUSS,
        tolerance: float = 1.0e-6,
        res_folder: str = None
    ):
        """

        Args:
            mesh_file_path:
            field:
            time_steps:
            iterations:
            finite_element:
            boundary_conditions:
            loads:
            quadrature_type:
            tolerance:
        """
        self.finite_element = finite_element
        self.field = field
        self.mesh = Mesh(mesh_file_path=mesh_file_path, integration_order=finite_element.construction_integration_order)
        self.__check_loads(loads)
        self.__check_boundary_conditions(boundary_conditions)
        self.boundary_conditions = boundary_conditions
        self.loads = loads
        self.time_steps = time_steps
        self.number_of_iterations = iterations
        self.tolerance = tolerance
        self.quadrature_type = quadrature_type
        # ------ build elements
        self.elements = self.get_elements()
        return

    def get_total_system_size(self) -> (int, int):
        """

        Returns:

        """
        constrained_faces = 0
        constrained_constants = 0
        for key, val in self.mesh.faces_boundaries_connectivity.items():
            for bc in self.boundary_conditions:
                if key == bc.boundary_name and bc.boundary_type == BoundaryType.DISPLACEMENT:
                    constrained_faces += len(val)
                elif key == bc.boundary_name and bc.boundary_type == BoundaryType.SLIDE:
                    constrained_constants += len(val)
        constrained_faces_matrix_size = constrained_faces * self.finite_element.face_basis_k.dimension
        constrained_const_matrix_size = constrained_constants
        lagrange_system_size = constrained_faces_matrix_size + constrained_const_matrix_size
        system_size = self.mesh.number_of_faces_in_mesh * self.finite_element.face_basis_k.dimension * self.field.field_dimension
        constrained_system_size = system_size + lagrange_system_size
        return constrained_system_size, system_size

    def create_vertex_res_files(self, suffix: str):
        """

        Args:
            suffix:

        Returns:

        """
        with open(os.path.join(get_project_path(), "res/res_vtx_{}.csv".format(suffix)), "w") as res_vtx_file:
            for x_dir in range(self.field.euclidean_dimension):
                res_vtx_file.write("X_{},".format(x_dir))
            for u_dir in range(self.field.field_dimension):
                res_vtx_file.write("{}_{},".format(self.field.label, u_dir))
            res_vtx_file.write("\n")
        return

    def create_quadrature_points_res_files(self, suffix: str, material: Material):
        """

        Args:
            suffix:
            material:
        """
        with open(os.path.join(get_project_path(), "res/res_qdp_{}.csv".format(suffix)), "w") as res_qdp_file:
            for x_dir in range(self.field.euclidean_dimension):
                res_qdp_file.write("XQ_{},".format(x_dir))
            for u_dir in range(self.field.field_dimension):
                res_qdp_file.write("{}_{},".format(self.field.label, u_dir))
            for strain_component in range(self.field.gradient_dimension):
                res_qdp_file.write("STRAIN_{},".format(strain_component))
            for stress_component in range(self.field.gradient_dimension):
                res_qdp_file.write("STRESS_{},".format(stress_component))
            res_qdp_file.write("STRAIN_TRACE,")
            res_qdp_file.write("HYDRO_STRESS,")
            if material.behaviour_name != "Elasticity":
                # try:
                isv = material.mat_data.s1.internal_state_variables
                for isv_val in range(len(isv[0])):
                    res_qdp_file.write("INTERNAL_STATE_VARIABLE_{},".format(isv_val))
            # except:
            #     pass
            # stored_energies
            # ', '
            # dissipated_energies
            # ', '
            # internal_state_variables
            # '
            # for
            res_qdp_file.write("\n")

    def write_vertex_res_files(self, suffix: str, faces_unknown_vector: ndarray):
        """

        Args:
            suffix:
            faces_unknown_vector:

        Returns:

        """
        with open(os.path.join(get_project_path(), "res/res_vtx_{}.csv".format(suffix)), "a") as res_vtx_file:
            for vertex_count in range(self.mesh.number_of_vertices_in_mesh):
                vertex = self.mesh.vertices[:, vertex_count]
                for x_dir in range(self.field.euclidean_dimension):
                    res_vtx_file.write("{},".format(vertex[x_dir]))
                vertex_field_value = np.zeros((self.field.field_dimension,), dtype=real)
                for c, cell_vertices_connectivity in enumerate(self.mesh.cells_vertices_connectivity):
                    if vertex_count in cell_vertices_connectivity:
                        for u_dir in range(self.field.field_dimension):
                            # vertex_field_value[u_dir] += self.elements[c].get_cell_field_increment_value(
                            #     point=vertex,
                            #     direction=u_dir,
                            #     field=self.field,
                            #     finite_element=self.finite_element,
                            #     element_unknown_vector=unknown_increment,
                            # )
                            vertex_field_value[u_dir] += self.elements[c].get_cell_field_value(
                                faces_unknown_vector=faces_unknown_vector, point=vertex, direction=u_dir,
                            )
                vertex_field_value = vertex_field_value / self.mesh.vertices_weights_cell[vertex_count]
                for u_dir in range(self.field.field_dimension):
                    res_vtx_file.write("{},".format(vertex_field_value[u_dir]))
                res_vtx_file.write("\n")
        return

    def write_quadrature_points_res_files(self, suffix: str, material: Material, faces_unknown_vector: ndarray):
        """

        Args:
            suffix:
            material:
            faces_unknown_vector:

        Returns:

        """
        with open(os.path.join(get_project_path(), "res/res_qdp_{}.csv".format(suffix)), "a") as res_qdp_file:
            qp = 0
            for element in self.elements:
                cell_quadrature_size = element.cell.get_quadrature_size(
                    element.finite_element.construction_integration_order
                )
                cell_quadrature_points = element.cell.get_quadrature_points(
                    element.finite_element.construction_integration_order
                )
                cell_quadrature_weights = element.cell.get_quadrature_weights(
                    element.finite_element.construction_integration_order
                )
                for qc in range(cell_quadrature_size):
                    x_q_c = cell_quadrature_points[:, qc]
                    for x_dir in range(self.field.euclidean_dimension):
                        res_qdp_file.write("{},".format(x_q_c[x_dir]))
                    for u_dir in range(self.field.field_dimension):
                        # quad_point_field_value = element.get_cell_field_increment_value(
                        #     point=x_q_c,
                        #     direction=u_dir,
                        #     field=self.field,
                        #     finite_element=self.finite_element,
                        #     element_unknown_vector=unknown_increment,
                        # )
                        quad_point_field_value = element.get_cell_field_value(
                            faces_unknown_vector=faces_unknown_vector, point=x_q_c, direction=u_dir,
                        )
                        res_qdp_file.write("{},".format(quad_point_field_value))
                    for g_dir in range(self.field.gradient_dimension):
                        strain_component = material.mat_data.s1.gradients[qp, g_dir]
                        res_qdp_file.write("{},".format(strain_component))
                    for g_dir in range(self.field.gradient_dimension):
                        if self.field.grad_type == GradType.DISPLACEMENT_TRANSFORMATION_GRADIENT:
                            if self.field.field_type in [FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN, FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRESS]:
                                F = np.zeros((3, 3), dtype=real)
                                F[0, 0] = material.mat_data.s1.gradients[qp, 0]
                                F[1, 1] = material.mat_data.s1.gradients[qp, 1]
                                F[2, 2] = material.mat_data.s1.gradients[qp, 2]
                                F[0, 1] = material.mat_data.s1.gradients[qp, 3]
                                F[1, 0] = material.mat_data.s1.gradients[qp, 4]
                                PK = np.zeros((3, 3), dtype=real)
                                PK[0, 0] = material.mat_data.s1.thermodynamic_forces[qp, 0]
                                PK[1, 1] = material.mat_data.s1.thermodynamic_forces[qp, 1]
                                PK[2, 2] = material.mat_data.s1.thermodynamic_forces[qp, 2]
                                PK[0, 1] = material.mat_data.s1.thermodynamic_forces[qp, 3]
                                PK[1, 0] = material.mat_data.s1.thermodynamic_forces[qp, 4]
                                J = np.linalg.det(F)
                                # F_T_inv = np.linalg.inv(F.T)
                                sig = (1.0 / J) * PK @ F.T
                                sig_vect = np.zeros((5,), dtype=real)
                                sig_vect[0] = sig[0, 0]
                                sig_vect[1] = sig[1, 1]
                                sig_vect[2] = sig[2, 2]
                                sig_vect[3] = sig[0, 1]
                                sig_vect[4] = sig[1, 0]
                                stress_component = sig_vect[g_dir]
                            elif self.field.field_type == FieldType.DISPLACEMENT_LARGE_STRAIN:
                                F = np.zeros((3, 3), dtype=real)
                                F[0, 0] = material.mat_data.s1.gradients[qp, 0]
                                F[1, 1] = material.mat_data.s1.gradients[qp, 1]
                                F[2, 2] = material.mat_data.s1.gradients[qp, 2]
                                F[0, 1] = material.mat_data.s1.gradients[qp, 3]
                                F[1, 0] = material.mat_data.s1.gradients[qp, 4]
                                F[0, 2] = material.mat_data.s1.gradients[qp, 5]
                                F[2, 0] = material.mat_data.s1.gradients[qp, 6]
                                F[1, 2] = material.mat_data.s1.gradients[qp, 7]
                                F[2, 1] = material.mat_data.s1.gradients[qp, 8]
                                PK = np.zeros((3, 3), dtype=real)
                                PK[0, 0] = material.mat_data.s1.thermodynamic_forces[qp, 0]
                                PK[1, 1] = material.mat_data.s1.thermodynamic_forces[qp, 1]
                                PK[2, 2] = material.mat_data.s1.thermodynamic_forces[qp, 2]
                                PK[0, 1] = material.mat_data.s1.thermodynamic_forces[qp, 3]
                                PK[1, 0] = material.mat_data.s1.thermodynamic_forces[qp, 4]
                                PK[0, 2] = material.mat_data.s1.thermodynamic_forces[qp, 5]
                                PK[2, 0] = material.mat_data.s1.thermodynamic_forces[qp, 6]
                                PK[1, 2] = material.mat_data.s1.thermodynamic_forces[qp, 7]
                                PK[2, 1] = material.mat_data.s1.thermodynamic_forces[qp, 8]
                                J = np.linalg.det(F)
                                # F_T_inv = np.linalg.inv(F.T)
                                sig = (1.0 / J) * PK @ F.T
                                sig_vect = np.zeros((9,), dtype=real)
                                sig_vect[0] = sig[0, 0]
                                sig_vect[1] = sig[1, 1]
                                sig_vect[2] = sig[2, 2]
                                sig_vect[3] = sig[0, 1]
                                sig_vect[4] = sig[1, 0]
                                sig_vect[5] = sig[0, 2]
                                sig_vect[6] = sig[2, 0]
                                sig_vect[7] = sig[1, 2]
                                sig_vect[8] = sig[2, 1]
                                stress_component = sig_vect[g_dir]
                        elif self.field.grad_type == GradType.DISPLACEMENT_SMALL_STRAIN:
                            stress_component = material.mat_data.s1.thermodynamic_forces[qp, g_dir]
                        res_qdp_file.write("{},".format(stress_component))
                    hyrdostatic_pressure = 0.0
                    strain_trace = 0.0
                    if self.field.field_type in [
                        FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN,
                        FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRESS,
                        FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN,
                        FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRESS,
                    ]:
                        num_diagonal_components = 3
                    else:
                        num_diagonal_components = self.field.field_dimension
                    for x_dir in range(num_diagonal_components):
                        strain_trace += material.mat_data.s1.gradients[qp, x_dir]
                        hyrdostatic_pressure += material.mat_data.s1.thermodynamic_forces[qp, x_dir]
                    hyrdostatic_pressure = hyrdostatic_pressure / num_diagonal_components
                    res_qdp_file.write("{},".format(strain_trace))
                    res_qdp_file.write("{},".format(hyrdostatic_pressure))
                    # try:
                    #     isv = material.mat_data.s1.internal_state_variables[qp]
                    #     for isv_val in range(len(isv)):
                    #         res_qdp_file.write("{},".format(isv[isv_val]))
                    # except:
                    #     pass
                    if material.behaviour_name != "Elasticity":
                        isv = material.mat_data.s1.internal_state_variables[qp]
                        for isv_val in range(len(isv)):
                            res_qdp_file.write("{},".format(isv[isv_val]))
                    qp += 1
                    res_qdp_file.write("\n")
        return

    def get_elements(self):
        """

        Returns:

        """
        elements = []
        _fk = self.finite_element.face_basis_k.dimension
        _dx = self.field.field_dimension
        _cl = self.finite_element.cell_basis_l.dimension
        for cell_index in range(self.mesh.number_of_cells_in_mesh):
            cell_vertices_connectivity = self.mesh.cells_vertices_connectivity[cell_index]
            cell_faces_connectivity = self.mesh.cells_faces_connectivity[cell_index]
            cell_ordering = self.mesh.cells_ordering[cell_index]
            cell_shape_type = self.mesh.cells_shape_types[cell_index]
            cell_vertices = self.mesh.vertices[:, cell_vertices_connectivity]
            # element_cell = Cell(cell_shape_type, cell_vertices, integration_order, quadrature_type=quadrature_type)
            element_cell = Shape(cell_shape_type, cell_vertices, connectivity=cell_ordering)
            element_faces = []
            element_faces_indices = []
            for global_face_index in cell_faces_connectivity:
                element_faces_indices.append(global_face_index)
                face_vertices_indices = self.mesh.faces_vertices_connectivity[global_face_index]
                face_vertices = self.mesh.vertices[:, face_vertices_indices]
                face_shape_type = self.mesh.faces_shape_types[global_face_index]
                # face = Face(face_shape_type, face_vertices, integration_order, quadrature_type=quadrature_type)
                face = Shape(face_shape_type, face_vertices)
                element_faces.append(face)
            element = Element(
                self.field,
                self.finite_element,
                element_cell,
                element_faces,
                element_faces_indices,
            )
            elements.append(element)
            del element_cell
            del element_faces
        # constrained_system_size, system_size = self.get_total_system_size()
        # iter_face_constraint = 0
        # for boundary_condition in self.boundary_conditions:
        #     if boundary_condition.boundary_type == BoundaryType.DISPLACEMENT:
        #         for element in elements:
        #             for f_local, f_global in enumerate(element.faces_indices):
        #                 if f_global in self.mesh.faces_boundaries_connectivity[boundary_condition.boundary_name]:
        #                     _l0 = system_size + iter_face_constraint * _fk
        #                     _l1 = system_size + (iter_face_constraint + 1) * _fk
        #                     _c0 = _cl * _dx + (f_local * _dx * _fk) + boundary_condition.direction * _fk
        #                     _c1 = _cl * _dx + (f_local * _dx * _fk) + (boundary_condition.direction + 1) * _fk
        #                     _r0 = f_global * _fk * _dx + _fk * boundary_condition.direction
        #                     _r1 = f_global * _fk * _dx + _fk * (boundary_condition.direction + 1)
        #                     element.faces_lagrange_system_row_positions[f_local][boundary_condition.direction] = (
        #                         _l0,
        #                         _l1,
        #                     )
        #                     element.faces_lagrange_system_col_positions[f_local][boundary_condition.direction] = (
        #                         _r0,
        #                         _r1,
        #                     )
        #                     element.faces_lagrange_local_positions[f_local][boundary_condition.direction] = (_c0, _c1)
        #                     iter_face_constraint += 1
        return elements

    def __check_loads(self, loads: List[Load]):
        """

        Args:
            loads:

        Returns:

        """
        if loads is None:
            return
        if isinstance(loads, list):
            if self.field.field_dimension >= len(loads) > 0:
                for i in range(len(loads)):
                    if isinstance(loads[i], Load):
                        if loads[i].direction < self.field.field_dimension:
                            continue
                        else:
                            raise ValueError
                    else:
                        raise TypeError("loads must be a list of Load")
            else:
                ValueError("loads must be a list of Load of size =< {}".format(self.field.field_dimension))
        else:
            raise TypeError("loads must be a list of Load of size =< {}".format(self.field.field_dimension))
        return

    def __check_boundary_conditions(self, boundary_conditions: List[BoundaryCondition]):
        """

        Args:
            boundary_conditions:

        Returns:

        """
        if isinstance(boundary_conditions, list):
            for boundary_condition in boundary_conditions:
                if isinstance(boundary_condition, BoundaryCondition):
                    if boundary_condition.boundary_name in self.mesh.faces_boundaries_connectivity.keys():
                        if boundary_condition.direction < self.field.field_dimension:
                            continue
                        else:
                            raise ValueError
                    else:
                        raise KeyError
                else:
                    raise TypeError
        else:
            raise TypeError
        return
