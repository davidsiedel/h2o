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
    def test_bande_perforee_small_strain_linear_elasticity(self):
        # --- VALUES
        # On détermine ici le tableau des pas de temps. Dans le contexte de la mécanique quasi-statique, on néglige les
        # effets d'accélération, et on peut se permettre de confondre le temps avec la variable de chargement (en
        # déplacement ou en pression). Le chargement de ce cas test est une traction uniaxiale imposée sur la face
        # supérieure de la bande, on définit donc l'intensité de la force de traction finale u_max, et initiale u_min
        # afin de définir N pas de temps uniformément répartis entre u_min et u_max, avec N le nombre de pas de temps.

        u_min = 0.0
        u_max = 0.0013
        time_steps = np.linspace(u_min, u_max, 10)
        iterations = 10

        # --- LOAD
        #On définit les conditions de chargement volumiques; il s'agit d'une fonction de l'espace et du temps, que l'on
        # passe en argument au modèle. Dans le cas de la bande perforée, on néglige l'influence du poids propre de la
        # structure devant celle des conditions aux limites; on donne donc une charge volumique nulle, pour tout instant
        # et tout point de la structure.
        def volumetric_load(time: float, position: ndarray):
            return 0

        loads = [Load(volumetric_load, 0), Load(volumetric_load, 1)]




        # --- BC
        # on definit les conditions limites du problème. la fonction pull qui prend en argument le temps retourne le temps
        # qui est confondu ici avec la variable de chargement. et la fonction fixed qui prend egalement en argument le temps
        # retourne une valeur nulle afin de bloquer les parties où l'on appliquera cette fonction.
        def pull(time: float, position: ndarray) -> float:
            return time

        def fixed(time: float, position: ndarray) -> float:
            return 0.0

        boundary_conditions = [
            BoundaryCondition("top", pull, BoundaryType.DISPLACEMENT, 1),       # applique l'extension sur le haut de la structure selon y
            BoundaryCondition("bottom", fixed, BoundaryType.DISPLACEMENT, 1),   # bloque la bas selon y
            BoundaryCondition("left", fixed, BoundaryType.DISPLACEMENT, 0),     # bloque la gauche selon x
        ]

        # --- MESH
        # on introduit le maillage préalablement fait sur gmsh.
        #mesh_file_path = ("meshes/bande_2_bas_carre.msh"  )
        #mesh_file_path = ("meshes/bande_bas_carre.msh")
        #mesh_file_path = ("meshes/bande_triangulaire_gros.msh")
        mesh_file_path = ("meshes/bande_maillage_fin.msh")
        # --- FIELD
        # definition du type de deplacement
        displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN)


        # --- FINITE ELEMENT
        finite_element = FiniteElement(element_type=ElementType.HDG_EQUAL,polynomial_order=1,euclidean_dimension=displacement.euclidean_dimension,
            basis_type=BasisType.MONOMIAL)


        # --- PROBLEM------------introduction des données du problème
        p = Problem(mesh_file_path=mesh_file_path,field=displacement,finite_element=finite_element,time_steps=time_steps,
                    iterations=iterations,
            boundary_conditions=boundary_conditions,loads=loads,quadrature_type=QuadratureType.GAUSS,tolerance=1.0e-4,res_folder_path=get_current_res_folder_path())


        # --- MATERIAL----introduction des données matériau
        parameters = {"YoungModulus": 70e6, "PoissonRatio": 0.4999}
        stabilization_parameter = 1 *parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        mat = Material(nq=p.mesh.number_of_cell_quadrature_points_in_mesh,library_path="behaviour/src/libBehaviour.dylib",
            library_name="SmallStrainIsotropicLinearHardeningPlasticity",hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=stabilization_parameter,lagrange_parameter=parameters["YoungModulus"],field=displacement,parameters=None)


        # --- SOLVE------Resolution du probleme
        solve_newton_2(p, mat, verbose=False)
