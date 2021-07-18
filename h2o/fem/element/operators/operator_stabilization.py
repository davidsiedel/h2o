from h2o.h2o import *
from h2o.geometry.shape import Shape, get_rotation_matrix
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement

def get_stabilization_operator2(field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape]) -> ndarray:
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    _io = finite_element.construction_integration_order
    x_c = cell.get_centroid()
    h_c = cell.get_diameter()
    bdc = cell.get_bounding_box()
    stabilization_operator = np.zeros((_es, _es), dtype=real)
    stabilization_op = np.zeros((_fk * _dx, _es), dtype=real)
    for _f, face in enumerate(faces):
        h_f = face.get_diameter()
        bdf = face.get_bounding_box()
        x_f = face.get_centroid()
        face_quadrature_points = face.get_quadrature_points(_io)
        face_quadrature_weights = face.get_quadrature_weights(_io)
        face_quadrature_size = face.get_quadrature_size(_io)
        face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
        m_mas_f = np.zeros((_fk, _fk), dtype=real)
        m_hyb_f = np.zeros((_fk, _cl), dtype=real)
        for qf in range(face_quadrature_size):
            x_q_f = face_quadrature_points[:, qf]
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            bdf_proj = (face_rotation_matrix @ bdf)[:-1]
            bdf_proj = face.get_face_bounding_box()
            x_q_f_p = face_rotation_matrix @ x_q_f
            x_c_p = face_rotation_matrix @ x_c
            # phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, h_c)
            phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
            # phi_l = cell_basis_l.get_phi_vector(x_q_f_p, x_c_p, h_c)
            # psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, h_f)
            psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
            m_hyb_f += w_q_f * np.tensordot(psi_k, phi_l, axes=0)
            m_mas_f += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
        m_mas_f_inv = np.linalg.inv(m_mas_f)
        proj_mat = m_mas_f_inv @ m_hyb_f
        m = np.eye(_fk, dtype=real)
        m_mas_f2 = np.zeros((_dx * _fk, _dx * _fk), dtype=real)
        for _i in range(_dx):
            _ri = _i * _fk
            _rj = (_i + 1) * _fk
            _ci = _i * _cl
            _cj = (_i + 1) * _cl
            stabilization_op[_ri:_rj, _ci:_cj] -= proj_mat
            _ci = _dx * _cl + _f * _dx * _fk + _i * _fk
            _cj = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
            stabilization_op[_ri:_rj, _ci:_cj] += m
            m_mas_f2[_ri:_rj,_ri:_rj] += m_mas_f
        stabilization_operator += (1.0 / h_f) * stabilization_op.T @ m_mas_f2 @ stabilization_op
        # stabilization_operator += stabilization_op.T @ m_mas_f @ stabilization_op
    return stabilization_operator


def get_stabilization_operator3(field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape]) -> ndarray:
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    _io = finite_element.construction_integration_order
    x_c = cell.get_centroid()
    h_c = cell.get_diameter()
    bdc = cell.get_bounding_box()
    stabilization_operator = np.zeros((_es, _es), dtype=real)
    stabilization_op = np.zeros((_fk * _dx, _es), dtype=real)
    for _f, face in enumerate(faces):
        h_f = face.get_diameter()
        bdf = face.get_bounding_box()
        x_f = face.get_centroid()
        face_quadrature_points = face.get_quadrature_points(_io)
        face_quadrature_weights = face.get_quadrature_weights(_io)
        face_quadrature_size = face.get_quadrature_size(_io)
        face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
        m_mas_f = np.zeros((_fk, _fk), dtype=real)
        m_hyb_f = np.zeros((_fk, _cl), dtype=real)
        for qf in range(face_quadrature_size):
            x_q_f = face_quadrature_points[:, qf]
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            bdf_proj = (face_rotation_matrix @ bdf)[:-1]
            bdf_proj = face.get_face_bounding_box()
            x_q_f_p = face_rotation_matrix @ x_q_f
            x_c_p = face_rotation_matrix @ x_c
            # phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, h_c)
            phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
            # phi_l = cell_basis_l.get_phi_vector(x_q_f_p, x_c_p, h_c)
            # psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, h_f)
            psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
            m_hyb_f += w_q_f * np.tensordot(psi_k, phi_l, axes=0)
            m_mas_f += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
        m_mas_f_inv = np.linalg.inv(m_mas_f)
        proj_mat = m_mas_f_inv @ m_hyb_f
        m = np.eye(_fk, dtype=real)
        m_mas_f2 = np.zeros((_dx * _fk, _dx * _fk), dtype=real)
        for _i in range(_dx):
            stabilization_op33 = np.zeros((_fk, _es), dtype=real)
            _ri = _i * _fk
            _rj = (_i + 1) * _fk
            _ci = _i * _cl
            _cj = (_i + 1) * _cl
            stabilization_op[_ri:_rj, _ci:_cj] -= proj_mat
            stabilization_op33[:, _ci:_cj] -= proj_mat
            _ci = _dx * _cl + _f * _dx * _fk + _i * _fk
            _cj = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
            stabilization_op[_ri:_rj, _ci:_cj] += m
            stabilization_op33[:, _ci:_cj] += m
            m_mas_f2[_ri:_rj,_ri:_rj] += m_mas_f
            # stabilization_operator += (1.0 / h_f) * stabilization_op.T @ m_mas_f2 @ stabilization_op
            stabilization_operator += (1.0 / h_f) * stabilization_op33.T @ m_mas_f @ stabilization_op33
        # stabilization_operator += stabilization_op.T @ m_mas_f @ stabilization_op
    def get_f_corr(find: int)-> int:
        if find == 0:
            return 3
        elif find == 1:
            return 2
        elif find == 2:
            return 0
        elif find == 3:
            return 1
        else:
            return -1
    def get_correspondance(r: int) -> int:
        if r < _dx * _cl:
            c = 0
            for _row0 in range(_cl):
                for _row1 in range(_dx):
                    pos0 = _cl * _row1 + _row0
                    if r == pos0:
                        return c
                    c += 1
        elif r >= _dx * _cl:
            c = _dx * _cl
            for _f in range(len(faces)):
                for _row0 in range(_fk):
                    for _row1 in range(_dx):
                        pos00 = _fk * _row1 + _row0
                        f_pkr = get_f_corr(_f)
                        # pos0 = _dx * _cl + _f * _dx * _fk + pos00
                        pos0 = _dx * _cl + f_pkr * _dx * _fk + pos00
                        if r == pos0:
                            return c
                        c += 1
        else:
            return -1
    stb_print = np.zeros((_es, _es), dtype=real)
    for _row in range(_es):
        for _col in range(_es):
            rr = get_correspondance(_row)
            cc = get_correspondance(_col)
            # print(rr)
            # print("--> {}".format(_col))
            # print(cc)
            stb_print[rr, cc] = stabilization_operator[_row, _col]
    # pos1 = 0
    # for _row0 in range(_cl):
    #     for _row1 in range(_dx):
    #         pos0 = _cl * _row1 + _row0
    #         print("pos0 : {}".format(pos0))
    #         print("pos1 : {}".format(pos1))
    #         # stb_print[pos1 ,pos1] = stabilization_operator[pos0,pos0]
    #         col1 = 0
    #         for _col0 in range(_cl):
    #             for _col1 in range(_dx):
    #                 col0 = _cl * _col1 + _col0
    #                 stb_print[pos1, col1] = stabilization_operator[pos0, col0]
    #                 col1 += 1
    #         pos1 += 1
    # for _f in range(len(faces)):
    #     for _row0 in range(_fk):
    #         for _row1 in range(_dx):
    #             pos00 = _fk * _row1 + _row0
    #             pos0 = _dx * _cl + _f * _dx * _fk + pos00
    #             print("pos0 : {}".format(pos0))
    #             print("pos1 : {}".format(pos1))
    #             # stb_print[pos1 ,pos1] = stabilization_operator[pos0,pos0]
    #             col1 = 0
    #             for _f in range(len(faces)):
    #                 for _col0 in range(_fk):
    #                     for _col1 in range(_dx):
    #                         col00 = _fk * _col1 + _col0
    #                         col0 = _dx * _cl + _f * _dx * _fk + col00
    #                         stb_print[pos1, col1] = stabilization_operator[pos0, col0]
    #                         col1 += 1
    #             pos1 += 1
        #
        # for _col in range(_es):
        #     stb_print[_col, _row] = stabilization_operator[_row, _col]
    # print(stabilization_operator)
    if debug_mode == DebugMode.LIGHT:
        print("STAB")
        print(stb_print)
    return stabilization_operator


def get_stabilization_operator_component(
    field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], _f: int, _i: int
) -> ndarray:
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    _io = finite_element.construction_integration_order
    x_c = cell.get_centroid()
    h_c = cell.get_diameter()
    bdc = cell.get_bounding_box()
    stabilization_operator = np.zeros((_es, _es), dtype=real)
    stabilization_op = np.zeros((_fk, _es), dtype=real)
    face = faces[_f]
    h_f = face.get_diameter()
    x_f = face.get_centroid()
    bdf = face.get_bounding_box()
    face_quadrature_points = face.get_quadrature_points(_io)
    face_quadrature_weights = face.get_quadrature_weights(_io)
    face_quadrature_size = face.get_quadrature_size(_io)
    face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
    m_mas_f = np.zeros((_fk, _fk), dtype=real)
    m_hyb_f = np.zeros((_fk, _cl), dtype=real)
    for qf in range(face_quadrature_size):
        x_q_f = face_quadrature_points[:, qf]
        w_q_f = face_quadrature_weights[qf]
        s_f = (face_rotation_matrix @ x_f)[:-1]
        s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
        bdf_proj = (face_rotation_matrix @ bdf)[:-1]
        bdf_proj = face.get_face_bounding_box()
        x_q_f_p = face_rotation_matrix @ x_q_f
        x_c_p = face_rotation_matrix @ x_c
        # phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, h_c)
        phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
        # phi_l = cell_basis_l.get_phi_vector(x_q_f_p, x_c_p, h_c)
        # psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, h_f)
        psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
        m_hyb_f += w_q_f * np.tensordot(psi_k, phi_l, axes=0)
        m_mas_f += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
    m_mas_f_inv = np.linalg.inv(m_mas_f)
    proj_mat = m_mas_f_inv @ m_hyb_f
    m = np.eye(_fk, dtype=real)
    _ci = _i * _cl
    _cj = (_i + 1) * _cl
    stabilization_op[:, _ci:_cj] -= proj_mat
    _ci = _dx * _cl + _f * _dx * _fk + _i * _fk
    _cj = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
    stabilization_op[:, _ci:_cj] += m
    # stabilization_operator += (1.0 / h_f) * stabilization_op.T @ m_mas_f @ stabilization_op
    stabilization_operator += stabilization_op.T @ m_mas_f @ stabilization_op
    return stabilization_operator


def get_stabilization_operator(field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape]) -> ndarray:
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    stabilization_operator = np.zeros((_es, _es), dtype=real)
    for _f, face in enumerate(faces):
        h_f = face.get_diameter()
        for _i in range(_dx):
            stabilization_operator += get_stabilization_operator_component(field, finite_element, cell, faces, _f, _i)
        stabilization_operator *= 1.0 / h_f
    if debug_mode == DebugMode.LIGHT:
        print("STAB")
        print(stabilization_operator)
    return stabilization_operator


def get_stab_test(field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape])-> ndarray:
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    stabilization_operator = np.zeros((_es, _es), dtype=real)
    h_c = cell.get_diameter()
    x_c = cell.get_centroid()
    bdc = cell.get_bounding_box()
    for _f, face in enumerate(faces):
        face_stabilization_operator = np.zeros((_es, _es), dtype=real)
        stabilization_vector_operator = np.zeros((_dx, _es), dtype=real)
        x_f = face.get_centroid()
        h_f = face.get_diameter()
        bdf = face.get_bounding_box()
        m_mas_f = np.zeros((_fk, _fk), dtype=real)
        m_hyb_f = np.zeros((_fk, _cl), dtype=real)
        # m_prj_f = np.zeros((_fk, _cl))
        face_rotation_matrix = face.get_rotation_matrix()
        face_quadrature_size = face.get_quadrature_size(finite_element.construction_integration_order)
        face_quadrature_points = face.get_quadrature_points(finite_element.construction_integration_order)
        face_quadrature_weights = face.get_quadrature_weights(finite_element.construction_integration_order)
        # for qf in range(len(face.quadrature_weights)):
        for qf in range(face_quadrature_size):
            x_q_f = face_quadrature_points[:, qf]
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            bdf_proj = (face_rotation_matrix @ bdf)[:-1]
            bdf_proj = face.get_face_bounding_box()
            # psi_k = w_q_f * finite_element.face_basis_k.evaluate_function(s_q_f, s_f, h_f)
            # phi_l = w_q_f * finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, h_c)
            # psi_k = w_q_f * finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
            # phi_l = w_q_f * finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
            psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
            phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
            m_mas_f += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
            m_hyb_f += w_q_f * np.tensordot(psi_k, phi_l, axes=0)
            # m_mas_f += get_face_mass_matrix_in_face(
            #     face, finite_element.face_basis_k, finite_element.face_basis_k, x_q_f, w_q_f
            # )
            # m_hyb_f += get_test_mass_matrix_in_face(
            #     cell, face, finite_element.cell_basis_l, finite_element.face_basis_k, x_q_f, w_q_f
            # )
        m_mas_f_inv = np.linalg.inv(m_mas_f)
        if debug_mode == DebugMode.LIGHT:
            print("MASS")
            print(m_mas_f)
        # if  == DebugMode.LIGHT:
        #     print("FACE MASS MATRIX IN STABILIZATION COND :")
        #     print("{}".format(np.linalg.cond(m_mas_f)))
        m_prj_f = m_mas_f_inv @ m_hyb_f
        m_eye_f = np.eye(_fk, dtype=real)
        for _x in range(_dx):
            stabilization_vector_component_op = np.zeros((_fk, _es), dtype=real)
            c0 = _x * _cl
            c1 = (_x + 1) * _cl
            stabilization_vector_component_op[:,c0:c1] -= m_prj_f
            c0 = _dx * _cl + _f * _dx * _fk + _x * _fk
            c1 = _dx * _cl + _f * _dx * _fk + (_x + 1) * _fk
            stabilization_vector_component_op[:,c0:c1] += m_eye_f
            # for qf in range(len(face.quadrature_weights)):
            for qf in range(face_quadrature_size):
                x_q_f = face_quadrature_points[:, qf]
                w_q_f = face_quadrature_weights[qf]
                s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
                bdf_proj = (face_rotation_matrix @ bdf)[:-1]
                bdf_proj = face.get_face_bounding_box()
                s_f = (face_rotation_matrix @ x_f)[:-1]
                # v_face = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, h_f)
                v_face = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
                # stabilization_vector_component_at_quad = v_face @ stabilization_vector_component_op
                stabilization_vector_operator[_x,:] += v_face @ stabilization_vector_component_op
        # for qf in range(len(face.quadrature_weights)):
        if debug_mode == DebugMode.LIGHT:
            print("stabilization_vector_component_op")
            print(stabilization_vector_component_op)
        for qf in range(face_quadrature_size):
            x_q_f = face_quadrature_points[:, qf]
            w_q_f = face_quadrature_weights[qf]
            m_eye_tan = np.eye(_dx, dtype=real)
            face_stabilization_operator += w_q_f * stabilization_vector_operator.T @ m_eye_tan @ stabilization_vector_operator
        weighted_face_stabilization_operator = (1.0/h_f) * face_stabilization_operator
        stabilization_operator += weighted_face_stabilization_operator
    if debug_mode == DebugMode.LIGHT:
        print("STAB")
        print(stabilization_operator)
    return stabilization_operator
