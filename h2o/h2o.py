from enum import Enum, auto
import numpy as np
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
    DISPLACEMENT = "DISPLACEMENT"
    PRESSURE = "PRESSURE"
    SLIDE = "SLIDE"


class ShapeType(Enum):
    SEGMENT = "SEGMENT"
    TRIANGLE = "TRIANGLE"


class QuadratureType(Enum):
    GAUSS = "GAUSS"


class BasisType(Enum):
    MONOMIAL = "MONOMIAL"


class ElementType(Enum):
    HDG_HIGH = "HDG_HIGH"
    HHO_HIGH = "HHO_HIGH"
    HDG_EQUAL = "HDG_EQUAL"
    HHO_EQUAL = "HHO_EQUAL"
    HDG_LOW = "HDG_LOW"
    HHO_LOW = "HHO_LOW"


class FieldType(Enum):
    SCALAR = "SCALAR"
    VECTOR = "VECTOR"
    DISPLACEMENT = "DISPLACEMENT"
    DISPLACEMENT_PLANE_STRAIN = "DISPLACEMENT_PLANE_STRAIN"
    DISPLACEMENT_PLANE_STRESS = "DISPLACEMENT_PLANE_STRAIN"


class StrainTempType(Enum):
    SMALL_STRAIN = ("SMALL_STRAIN",)
    FINITE_STRAIN = ("FINITE_STRAIN",)

class StressType(Enum):
    PIOLA_KIRCHOFF_1 = "PK1"
    CAUCHY = "CAUCHY"

class StrainType(Enum):
    DISPLACEMENT_TRANSFORMATION_GRADIENT = "F"
    DISPLACEMENT_SYMMETRIC_GRADIENT = "GRAD_SYM_U"


class DerivationType(Enum):
    SYMMETRIC = "SYMMETRIC"
    FULL = "FULL"

def get_project_path():
    return pathlib.Path(__file__).parent.parent.absolute()

def get_res_file_path(res_file_name: str, suffix: str):
    project_path = get_project_path()
    return os.path.join(project_path, "res/{}_{}.txt".format(res_file_name, suffix))