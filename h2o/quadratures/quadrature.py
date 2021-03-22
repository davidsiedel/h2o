import h2o.quadratures.gauss.gauss_segment as gauss_segment
import h2o.quadratures.gauss.gauss_triangle as gauss_triangle
import h2o.quadratures.gauss.gauss_quadrangle as gauss_quadrangle
import h2o.quadratures.gauss.gauss_tetrahedron as gauss_tetrahedron
import h2o.quadratures.gauss.gauss_hexahedron as gauss_hexahedron
import h2o.geometry.shapes.shape_segment as shape_segment
from h2o.quadratures.quadrature_utils import *


def get_quadrature_item(
    shape_type: ShapeType,
    integration_order: int,
    quadrature_item: QuadratureItem,
    quadrature_type: QuadratureType = QuadratureType.GAUSS,
) -> Union[ndarray, int]:
    """

    Args:
        shape_type:
        integration_order:
        quadrature_item:
        quadrature_type:

    Returns:

    """
    if quadrature_type == QuadratureType.GAUSS:
        if shape_type == ShapeType.SEGMENT:
            if quadrature_item in [QuadratureItem.POINTS, QuadratureItem.WEIGHTS, QuadratureItem.JACOBIAN]:
                return gauss_segment.get_segment_quadrature(integration_order, quadrature_item)
            elif quadrature_item == QuadratureItem.SIZE:
                return gauss_segment.get_number_of_quadrature_points_in_segment(integration_order)
        elif shape_type == ShapeType.TRIANGLE:
            if quadrature_item in [QuadratureItem.POINTS, QuadratureItem.WEIGHTS, QuadratureItem.JACOBIAN]:
                return gauss_triangle.get_reference_triangle_quadrature_item(integration_order, quadrature_item)
            elif quadrature_item == QuadratureItem.SIZE:
                return gauss_triangle.get_number_of_quadrature_points_in_triangle(integration_order)
        elif shape_type == ShapeType.QUADRANGLE:
            if quadrature_item in [QuadratureItem.POINTS, QuadratureItem.WEIGHTS, QuadratureItem.JACOBIAN]:
                return gauss_quadrangle.get_quadrangle_quadrature(integration_order, quadrature_item)
            elif quadrature_item == QuadratureItem.SIZE:
                return gauss_quadrangle.get_number_of_quadrature_points_in_quadrangle(integration_order)
        elif shape_type == ShapeType.TETRAHEDRON:
            if quadrature_item in [QuadratureItem.POINTS, QuadratureItem.WEIGHTS, QuadratureItem.JACOBIAN]:
                return gauss_tetrahedron.get_reference_tetrahedron_quadrature_item(integration_order, quadrature_item)
            elif quadrature_item == QuadratureItem.SIZE:
                return gauss_tetrahedron.get_number_of_quadrature_points_in_tetrahedron(integration_order)
        elif shape_type == ShapeType.HEXAHEDRON:
            if quadrature_item in [QuadratureItem.POINTS, QuadratureItem.WEIGHTS, QuadratureItem.JACOBIAN]:
                return gauss_hexahedron.get_reference_hexahedron_quadrature_item(integration_order, quadrature_item)
            elif quadrature_item == QuadratureItem.SIZE:
                return gauss_hexahedron.get_number_of_quadrature_points_in_hexahedron(integration_order)
        else:
            raise KeyError("unsupported shape")
    else:
        raise KeyError("unsupported quadrature type")


def get_shape_quadrature_points(
    shape_type: ShapeType,
    shape_vertices: ndarray,
    integration_order: int,
    quadrature_type: QuadratureType = QuadratureType.GAUSS,
) -> ndarray:
    """

    Args:
        shape_type:
        shape_vertices:
        integration_order:
        quadrature_type:

    Returns:

    """
    quadrature_reference_points = get_quadrature_item(
        shape_type, integration_order, QuadratureItem.POINTS, quadrature_type=quadrature_type
    )
    quadrature_points = (quadrature_reference_points @ shape_vertices.T).T
    return quadrature_points


def get_shape_quadrature_weights(
    shape_type: ShapeType,
    shape_vertices: ndarray,
    integration_order: int,
    quadrature_type: QuadratureType = QuadratureType.GAUSS,
) -> ndarray:
    """

    Args:
        shape_type:
        shape_vertices:
        integration_order:
        quadrature_type:

    Returns:

    """
    quadrature_reference_weights = get_quadrature_item(
        shape_type, integration_order, QuadratureItem.WEIGHTS, quadrature_type=quadrature_type
    )
    quadrature_jacobian = get_quadrature_item(
        shape_type, integration_order, QuadratureItem.JACOBIAN, quadrature_type=quadrature_type
    )
    quadrature_size = get_quadrature_item(
        shape_type, integration_order, QuadratureItem.SIZE, quadrature_type=quadrature_type
    )
    wgts = np.zeros((quadrature_size,), dtype=real)
    if shape_type == ShapeType.SEGMENT:
        rotation_matrix = shape_segment.get_segement_rotation_matrix(shape_vertices)
        projected_shape_vertices = (rotation_matrix @ shape_vertices)[:-1]
        # rotated_shape_vertices_2 = projected_shape_vertices[:-1]
        for i, jaco in enumerate(quadrature_jacobian):
            # wgts[i] = jaco[0] @ shape_vertices[0,:]
            wgts[i] = jaco[0] @ projected_shape_vertices[0,:]
    elif shape_type in [ShapeType.TRIANGLE, ShapeType.QUADRANGLE]:
        for i, jaco in enumerate(quadrature_jacobian):
            jac = np.zeros((2,2), dtype=real)
            jac[0,0] = jaco[0] @ shape_vertices[0,:]
            jac[0,1] = jaco[1] @ shape_vertices[0,:]
            jac[1,0] = jaco[2] @ shape_vertices[1,:]
            jac[1,1] = jaco[3] @ shape_vertices[1,:]
            wgts[i] = np.linalg.det(jac)
    elif shape_type in [ShapeType.TETRAHEDRON, ShapeType.HEXAHEDRON]:
        # coef = (1./6.)**(1./3.)
        # coef = 1.
        for i, jaco in enumerate(quadrature_jacobian):
            jac = np.zeros((3,3), dtype=real)
            jac[0,0] = jaco[0] @ shape_vertices[0,:]
            jac[0,1] = jaco[1] @ shape_vertices[0,:]
            jac[0,2] = jaco[2] @ shape_vertices[0,:]
            jac[1,0] = jaco[3] @ shape_vertices[1,:]
            jac[1,1] = jaco[4] @ shape_vertices[1,:]
            jac[1,2] = jaco[5] @ shape_vertices[1,:]
            jac[2,0] = jaco[6] @ shape_vertices[2,:]
            jac[2,1] = jaco[7] @ shape_vertices[2,:]
            jac[2,2] = jaco[8] @ shape_vertices[2,:]
            wgts[i] = np.linalg.det(jac)
    else:
        raise TypeError("No such type")
    # if shape_type == ShapeType.QUADRANGLE:
    #     coef = np.sqrt(4.)
    #     coef = 1.
    #     # coef = 1./np.sqrt(4.)
    #     # coef = 4.
    # elif shape_type == ShapeType.TRIANGLE:
    #     coef = np.sqrt(1./2)
    #     coef = 1.
    # for i, jaco in enumerate(quadrature_jacobian):
    #     jac = np.zeros((2,2), dtype=real)
    #     jac[0,0] = jaco[0] @ shape_vertices[0]
    #     jac[0,1] = jaco[1] @ shape_vertices[0]
    #     jac[1,0] = jaco[2] @ shape_vertices[1]
    #     jac[1,1] = jaco[3] @ shape_vertices[1]
    #     # wgts[i] = np.linalg.det(jac)
    #     # wgts[i] = coef * np.linalg.det(jac)
    #     wgts[i] = np.linalg.det(coef * jac)
    quadrature_weights = quadrature_reference_weights * wgts
    # quadrature_weights = wgts
    return quadrature_weights


def get_shape_quadrature_size(
    shape_type: ShapeType, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
) -> int:
    """

    Args:
        shape_type:
        integration_order:
        quadrature_type:

    Returns:

    """
    quadrature_size = get_quadrature_item(
        shape_type, integration_order, QuadratureItem.SIZE, quadrature_type=quadrature_type
    )
    return quadrature_size