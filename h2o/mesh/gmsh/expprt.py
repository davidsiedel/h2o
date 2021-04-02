import numpy as np

# from typing import Union, List, Tuple
# from dataclasses import dataclass

from data import *


@dataclass(frozen=True)
class PhysicalName:
    dim: int
    tag: int
    label: str


@dataclass(frozen=True)
class DomainEntity:
    dtype: DomainType
    tag: int
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    phys_tags: Union[List[int], None]
    bounding_entities: Union[List[int], None]


@dataclass(frozen=True)
class DataStructure:
    num_points: int
    num_curves: int
    num_surfaces: int
    num_volumes: int
    points_data: Union[List[DomainEntity], None]
    curves_data: Union[List[DomainEntity], None]
    surfaces_data: Union[List[DomainEntity], None]
    volumes_data: Union[List[DomainEntity], None]


@dataclass(frozen=True)
class Node:
    entity_dim: int
    entity_tag: int
    tag: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ElementEntity:
    entity_dim: int
    entity_tag: int
    element_type: int
    tag: int
    vertices_connectivity: List[int]


def get_domain_data(
    num_entities: int, line_index: int, c: List[str], domain_type: DomainType
) -> (Union[List[DomainEntity], None], int):
    if num_entities > 0:
        if domain_type != DomainType.POINT:
            domains_data = []
            for l_count in range(num_entities):
                line_index += 1
                line = c[line_index].split(" ")
                tag = int(line[0])
                min_p_x = float(line[1])
                min_p_y = float(line[2])
                min_p_z = float(line[3])
                max_p_x = float(line[4])
                max_p_y = float(line[5])
                max_p_z = float(line[6])
                num_physical_tags = int(line[7])
                offset = 7 + 1
                if num_physical_tags > 0:
                    physical_tags = []
                    for i_loc in range(num_physical_tags):
                        physical_tag = int(line[offset + i_loc])
                        physical_tags.append(physical_tag)
                else:
                    physical_tags = None
                num_bounding_curves = int(line[offset + num_physical_tags])
                offset += num_physical_tags + 1
                if num_bounding_curves > 0:
                    bounding_curves = []
                    for i_loc in range(num_bounding_curves):
                        bounding_curve = int(line[offset + i_loc])
                        bounding_curves.append(bounding_curve)
                else:
                    bounding_curves = None
                se = DomainEntity(
                    domain_type, tag, min_p_x, min_p_y, min_p_z, max_p_x, max_p_y, max_p_z, physical_tags, bounding_curves
                )
                domains_data.append(se)
        elif domain_type == DomainType.POINT:
            domains_data = []
            for p_count in range(num_entities):
                line_index += 1
                line = c[line_index].split(" ")
                tag = int(line[0])
                p_x = float(line[1])
                p_y = float(line[2])
                p_z = float(line[3])
                num_physical_tags = int(line[4])
                if num_physical_tags > 0:
                    physical_tags = []
                    for i_loc in range(num_physical_tags):
                        physical_tag = int(line[5 + i_loc])
                        physical_tags.append(physical_tag)
                else:
                    physical_tags = None
                se = DomainEntity(domain_type, tag, p_x, p_y, p_z, p_x, p_y, p_z, physical_tags, None)
                domains_data.append(se)
        else:
            raise ValueError("NO")
    else:
        domains_data = None
    return domains_data, line_index


def get_problem_euclidean_dimension(
    num_curves: int,
    num_surfaces: int,
    num_volumes: int,
) -> int:
    if num_curves == 0 and num_surfaces == 0 and num_volumes == 0:
        raise ValueError("NO")
    elif num_curves > 0 and num_surfaces == 0 and num_volumes == 0:
        return 1
    elif num_curves > 0 and num_surfaces > 0 and num_volumes == 0:
        return 2
    elif num_curves > 0 and num_surfaces > 0 and num_volumes > 0:
        return 3
    else:
        raise ValueError("NO")


def read_msh_file(msh_file_path: str) -> (DataStructure, List[Node], List[ElementEntity], int):
    with open(msh_file_path, "r") as msh_file:
        # --- READ MESH FILE
        c = msh_file.readlines()
        line_index = 4
        # --- PHYSICAL NAMES
        num_physical_entities = int(c[line_index])
        physical_entities_list = []
        for i in range(num_physical_entities):
            line = c[i + line_index + 1].rstrip().split(" ")
            dim = int(line[0])
            tag = int(line[1])
            label = str(line[2])
            pe = PhysicalName(dim, tag, label)
            physical_entities_list.append(pe)
        line_index += num_physical_entities
        offset = 3
        line_index += offset
        line = c[line_index].split(" ")
        # --- ENTITIES ENUMERATION
        num_points = int(line[0])
        num_curves = int(line[1])
        num_surfaces = int(line[2])
        num_volumes = int(line[3])
        euclidean_dimension = get_problem_euclidean_dimension(num_curves, num_surfaces, num_volumes)
        # --- POINTS, CURVES, SURFACES AND VOLUMES
        points_data, line_index = get_domain_data(num_points, line_index, c, DomainType.POINT)
        curves_data, line_index = get_domain_data(num_curves, line_index, c, DomainType.CURVE)
        surfaces_data, line_index = get_domain_data(num_surfaces, line_index, c, DomainType.SURFACE)
        volumes_data, line_index = get_domain_data(num_volumes, line_index, c, DomainType.VOLUME)
        # --- STRUCTURE
        ds = DataStructure(
            num_points,
            num_curves,
            num_surfaces,
            num_volumes,
            points_data,
            curves_data,
            surfaces_data,
            volumes_data,
        )
        # --- $NODES
        line_index += 3
        line = c[line_index].rstrip().split(" ")
        num_entity_blocks = int(line[0])
        num_nodes = int(line[1])
        min_node_tag = int(line[2])
        max_node_tag = int(line[3])
        # vertices = np.zeros((euclidean_dimension, num_nodes))
        nodes = []
        # ------------------------------------------------
        for entity_count in range(num_entity_blocks):
            line_index += 1
            line = c[line_index].rstrip().split(" ")
            entity_dim = int(line[0])
            entity_tag = int(line[1])
            param = int(line[2])
            nb_nodes_in_block = int(line[3])
            nodes_tags = np.zeros((nb_nodes_in_block,), dtype=int)
            for i_loc in range(nb_nodes_in_block):
                line_index += 1
                line = c[line_index].rstrip()
                loc_tag = int(line) - 1
                nodes_tags[i_loc] = loc_tag
            for i_loc in range(nb_nodes_in_block):
                line_index += 1
                line = c[line_index].rstrip().split(" ")
                x_pos = float(line[0])
                y_pos = float(line[1])
                z_pos = float(line[2])
                # vertices[:, nodes_tags[i_loc]] = np.array([x_pos, y_pos, z_pos][:euclidean_dimension])
                node = Node(entity_dim, entity_tag, nodes_tags[i_loc], x_pos, y_pos, z_pos)
                nodes.append(node)
        # --- $ELEMENTS
        line_index += 3
        line = c[line_index].rstrip().split(" ")
        num_entity_blocks = int(line[0])
        num_elements = int(line[1])
        min_element_tag = int(line[2])
        max_element_tag = int(line[3])
        element_entities = []
        # ------------------------------------------------
        for entity_count in range(num_entity_blocks):
            line_index += 1
            line = c[line_index].rstrip().split(" ")
            entity_dim = int(line[0])
            entity_tag = int(line[1])
            element_type = int(line[2])
            nb_elems_in_block = int(line[3])
            elems_tags = np.zeros((nb_elems_in_block,), dtype=int)
            for i_loc in range(nb_elems_in_block):
                line_index += 1
                line = c[line_index].rstrip().split(" ")
                loc_tag = int(line[0]) - 1
                elems_tags[i_loc] = loc_tag
                element_nb_nodes = get_element_data(element_type).n_nodes
                elems_vertices_connectivity = np.zeros((element_nb_nodes,), dtype=int)
                for v_count in range(element_nb_nodes):
                    elems_vertices_connectivity[v_count] = int(line[v_count + 1])
                ee = ElementEntity(entity_dim, entity_tag, element_type, loc_tag, list(elems_vertices_connectivity))
                element_entities.append(ee)
        # for ee_item in element_entities:
        #     print("--")
        #     print(ee_item.tag)
        #     print(ee_item.entity_dim)
        #     print(ee_item.entity_tag)
        #     # print(corr[ee_item.element_type][0])
        #     print("ee_item.element_type : {}".format(ee_item.element_type))
        #     print(get_element_data(ee_item.element_type))
        #     print(ee_item.vertices_connectivity)
        # for nd_item in nodes:
        #     print("--")
        #     print(nd_item.tag)
        #     print(nd_item.entity_tag)
        #     print(nd_item.entity_dim)
        #     print(nd_item.x)
        #     print(nd_item.y)
        #     print(nd_item.z)
        # for phys_ent in physical_entities_list:
        #     print("--")
        #     print(phys_ent.tag)
        #     print(phys_ent.dim)
        #     print(phys_ent.label)
        # # print(ds.volumes_data[0].phys_tags)
        # print("--")
        # print(vertices)
        return ds, nodes, element_entities, euclidean_dimension


read_msh_file("tetrahedra_1.msh")
# read_msh_file("quadrangles_0.msh")
# read_msh_file("triangles_0.msh")



