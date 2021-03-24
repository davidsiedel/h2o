from enum import Enum, auto
from typing import List, Dict, Tuple, Callable, Union
import numpy as np
from numpy import ndarray
import pathlib
import shutil
import os

# real = np.float64
real = float
# real = np.float32
# intg = np.uint8
intg = int
size_type = np.uint8

debug_mode = -1


class BoundaryType(Enum):
    DISPLACEMENT = auto()
    PRESSURE = auto()
    SLIDE = auto()


class ShapeType(Enum):
    SEGMENT = auto()
    TRIANGLE = auto()
    QUADRANGLE = auto()
    POLYGON = auto()
    TETRAHEDRON = auto()
    HEXAHEDRON = auto()
    POLYHEDRON = auto()


class QuadratureType(Enum):
    GAUSS = auto()


class BasisType(Enum):
    MONOMIAL = auto()


class ElementType(Enum):
    HDG_LOW = auto()
    HDG_EQUAL = auto()
    HDG_HIGH = auto()
    HHO_LOW = auto()
    HHO_EQUAL = auto()
    HHO_HIGH = auto()


class FieldType(Enum):
    DISPLACEMENT_LARGE_STRAIN = auto()
    DISPLACEMENT_SMALL_STRAIN = auto()
    DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN = auto()
    DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN = auto()
    DISPLACEMENT_LARGE_STRAIN_PLANE_STRESS = auto()
    DISPLACEMENT_SMALL_STRAIN_PLANE_STRESS = auto()


class FluxType(Enum):
    STRESS_PK1 = auto()
    STRESS_CAUCHY = auto()


class GradType(Enum):
    DISPLACEMENT_TRANSFORMATION_GRADIENT = auto()
    DISPLACEMENT_SMALL_STRAIN = auto()


class DerivationType(Enum):
    SYMMETRIC = auto()
    REGULAR = auto()


class GeometryError(Exception):
    pass


class QuadratureError(Exception):
    pass


def get_project_path():
    return pathlib.Path(__file__).parent.parent.absolute()


def get_res_file_path(res_file_name: str, suffix: str):
    project_path = get_project_path()
    return os.path.join(project_path, "res/{}_{}.txt".format(res_file_name, suffix))
