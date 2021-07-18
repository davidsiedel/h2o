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
    def test_cook_small_strain_linear_isotropic_hardening(self):
        # --- VALUES
        # On détermine ici le tableau des pas de temps. Dans le contexte de la mécanique quasi-statique, on néglige les
        # effets d'accélération, et on peut se permettre de confondre le temps avec la variable de chargement (en
        # déplacement ou en pression). Le chargement de ce cas test est une charge verticale imposée sur le coté droit
        # de la membrane, on définit donc l'intensité de la force  finale F_max, et initiale F_min
        # afin de définir N pas de temps uniformément répartis entre F_min et F_max, avec N le nombre de pas de temps.

        F_min= 0
        F_max= 5000000/0.016 # 1.8 newton/0.016 m2 (Pascal)
        time_steps = np.linspace(F_min, F_max, 25)
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
            BoundaryCondition("RIGHT", pull, BoundaryType.PRESSURE, 1),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
        ]

        # --- MESH
        #mesh_file_path = "meshes/cook_441Elem.msh"
        #mesh_file_path = "meshes/cook_961Elem.msh"
        mesh_file_path = "meshes/cook_taille_maille.msh"
        # --- FIELD
        displacement = Field(label="U", field_type= FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN)

        # --- FINITE ELEMENT
        finite_element = FiniteElement(
            element_type=ElementType.HDG_EQUAL,
            polynomial_order=3,
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
        parameters = {"YoungModulus": 206.9e9, "PoissonRatio": 0.29, "HardeningSlope": 0.135e6, "YieldStress": 0.243e6}
        # stabilization_parameter = 1000. * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        stabilization_parameter = 1 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        #stabilization_parameter = 1.0 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        mat = Material(
            nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
            library_path="behaviour/src/libBehaviour.dylib",
            library_name="Voce",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=stabilization_parameter,
            lagrange_parameter=parameters["YoungModulus"],
            field=displacement,
            parameters=None,
        )

        # --- SOLVE
        solve_newton_2(p, mat, verbose=False, debug_mode=DebugMode.NONE)
        # solve_newton_exact(p, mat, verbose=False, debug_mode=DebugMode.NONE)
