import torch
import numpy as np
import scipy
from .topology_utils import calculate_ve_mat, calculate_adj_matrix, calc_edges, calculate_incident_faces_to_edge, get_maximal_face_value_over_edges, calculate_vertex_incident_vector
from .geometry_utils import calculate_face_areas, calculate_vertex_normals, calc_edge_length,\
    calc_face_ref_normals, calc_face_normals
import torch
import torch.nn.functional as tfunc
import torch_scatter

A6000_hack = False

assert A6000_hack is not None, "assign True\False"


def prepend_dummies(
        vertices: torch.Tensor,  # V,D
        faces: torch.Tensor,  # F,3 long
):
    """prepend dummy elements to vertices and faces to enable "masked" scatter operations"""
    V, D = vertices.shape
    vertices = torch.concat((torch.full(
        (1, D), fill_value=torch.nan, device=vertices.device), vertices), dim=0)
    faces = torch.concat(
        (torch.zeros((1, 3), dtype=torch.long, device=faces.device), faces + 1), dim=0)
    return vertices, faces


def remove_dummies(
        vertices: torch.Tensor,  # V,D - first vertex all nan and unreferenced
        faces: torch.Tensor,  # F,3 long - first face all zeros
        face_colors=None
):
    if face_colors is not None:
        return vertices[1:], faces[1:] - 1, face_colors[1:]
    """remove dummy elements added with prepend_dummies()"""
    return vertices[1:], faces[1:] - 1, None

def pack(
        vertices: torch.Tensor,  # V,3 first unused and nan
        faces: torch.Tensor,  # F,3 long, 0 for unused
        return_old_to_new_face_id_map=False,
):  # (vertices,faces), keeps first vertex unused
    """removes unused elements in vertices and faces"""
    V = vertices.shape[0]
    F = faces.shape[0]

    # remove unused faces
    used_faces = faces[:, 0] != 0
    used_faces[0] = True
    faces = faces[used_faces]  # sync

    # face_idx_map - (-1) means face no longer exists
    old_to_new_face_idx_map = torch.zeros(F, device=vertices.device, dtype=torch.long)
    old_to_new_face_idx_map[used_faces] = torch.arange(faces.shape[0], device=vertices.device, dtype=torch.long)
    old_to_new_face_idx_map = old_to_new_face_idx_map - 1

    # remove unused vertices
    used_vertices = torch.zeros(V, 3, dtype=torch.bool, device=vertices.device)
    used_vertices.scatter_(dim=0, index=faces, value=True, reduce='add')  # TODO int faster?
    used_vertices = used_vertices.any(dim=1)
    used_vertices[0] = True
    vertices = vertices[used_vertices]  # sync

    # update used faces
    ind = torch.zeros(V, dtype=torch.long, device=vertices.device)
    V1 = used_vertices.sum()
    ind[used_vertices] = torch.arange(0, V1, device=vertices.device)  # sync
    faces = ind[faces]

    return vertices, faces, old_to_new_face_idx_map,used_vertices

def face_split(vs, face, face_mask, face_attribuates=None):
    """
    :param x: features B x V x C
    :param batch_tensors: SOA
    :param face_mask: binary mask where a face should be subdivided B x F
    :return: new features B X V' X C
    """
    device = vs.device
    face_mask_bool = face_mask.bool()
    faces_count = face.shape[0]
    vertices_count = vs.shape[0]
    new_face_count = faces_count + 2 * torch.sum(face_mask_bool)
    new_vertex_count = vertices_count + torch.sum(face_mask_bool)
    new_faces = torch.ones([new_face_count, 3], dtype=torch.int64, device=device)
    new_vertices = torch.ones([new_vertex_count, vs.shape[1]], dtype=torch.float, device=device)
    masked_faces = face[:faces_count][face_mask_bool]
    masked_face_count = masked_faces.shape[0]
    # update faces
    new_faces_mask = torch.ones_like(face_mask_bool, dtype=torch.int, device=device)
    new_faces_mask[face_mask_bool] = 3
    new_faces_mask = torch.cumsum(new_faces_mask, dim=0) - 1  # subtraction is to start at idx 0
    new_faces[new_faces_mask[~face_mask_bool]] = face[:faces_count][~face_mask_bool]
    new_face_ids = torch.arange(masked_face_count, device=device).unsqueeze(-1) + vertices_count
    new_faces[new_faces_mask[face_mask_bool] - 2] = torch.cat(
        [face[:faces_count, [0]][face_mask_bool], face[:faces_count, [1]][face_mask_bool], new_face_ids], dim=1)
    new_faces[new_faces_mask[face_mask_bool] - 1] = torch.cat(
        [face[:faces_count, [1]][face_mask_bool], face[:faces_count, [2]][face_mask_bool], new_face_ids], dim=1)
    new_faces[new_faces_mask[face_mask_bool] - 0] = torch.cat(
        [face[:faces_count, [2]][face_mask_bool], face[:faces_count, [0]][face_mask_bool], new_face_ids], dim=1)
    # update vertices
    new_vertices[:vs.shape[0]] = vs
    new_vertices[vs.shape[0]:] = torch.mean(vs[face[face_mask_bool]], dim=-2)
    new_face_ids = torch.stack([new_faces_mask[face_mask_bool] - 2,
                                new_faces_mask[face_mask_bool] - 1,
                                new_faces_mask[face_mask_bool] - 0], dim=-1)
    # create map from old to new face ids
    old_to_new_map = torch.arange(face.shape[0], device=device) + torch.cumsum(torch.roll(face_mask * 2, shifts=1),
                                                                               dim=0)
    return new_vertices, new_faces, new_face_ids.flatten(), old_to_new_map


def prepend_dummies(
        vertices: torch.Tensor,  # V,D
        faces: torch.Tensor,  # F,3 long
):
    """prepend dummy elements to vertices and faces to enable "masked" scatter operations"""
    V, D = vertices.shape
    vertices = torch.concat((torch.full(
        (1, D), fill_value=torch.nan, device=vertices.device), vertices), dim=0)
    faces = torch.concat(
        (torch.zeros((1, 3), dtype=torch.long, device=faces.device), faces + 1), dim=0)
    return vertices, faces

    
def rowwise_in(a, b):
    """
    a - tensor of size a0,c
    b - tensor of size b0,c
    returns - tensor of size a1 with 1 for each row of a in b, 0 otherwise
    """

    # dimensions
    shape1 = a.shape[0]
    shape2 = b.shape[0]
    c = a.shape[1]
    assert c == b.shape[1], "Tensors must have same number of columns"

    a_expand = a.unsqueeze(1).expand(-1, shape2, c)
    b_expand = b.unsqueeze(0).expand(shape1, -1, c)
    # element-wise equality
    return (a_expand == b_expand).all(-1).any(-1)


def split_edges(
        vertices: torch.Tensor,  # V,3 first unused
        faces: torch.Tensor,  # F,3 long, 0 for unused
        edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
        face_to_edge: torch.Tensor,  # F,3 long 0 for unused
        splits,  # E bool
        pack_faces: bool = True,
):  # (vertices,faces)

    #   c2                    c2               c...corners = faces
    #    . .                   . .             s...side_vert, 0 means no split
    #    .   .                 .N2 .           S...shrunk_face
    #    .     .               .     .         Ni...new_faces
    #   s2      s1           s2|c2...s1|c1
    #    .        .            .     .  .
    #    .          .          . S .      .
    #    .            .        . .     N1    .
    #   c0...(s0=0)....c1    s0|c0...........c1
    #
    # pseudo-code:
    #   S = [s0|c0,s1|c1,s2|c2] example:[c0,s1,s2]
    #   split = side_vert!=0 example:[False,True,True]
    #   N0 = split[0]*[c0,s0,s2|c2] example:[0,0,0]
    #   N1 = split[1]*[c1,s1,s0|c0] example:[c1,s1,c0]
    #   N2 = split[2]*[c2,s2,s1|c1] example:[c2,s2,s1]

    V = vertices.shape[0]
    F = faces.shape[0]
    S = splits.sum().item()  # sync

    if S == 0:
        return vertices, faces

    edge_vert = torch.zeros_like(splits, dtype=torch.long)  # E
    edge_vert[splits] = torch.arange(
        V, V + S, dtype=torch.long, device=vertices.device)  # E 0 for no split, sync
    side_vert = edge_vert[face_to_edge]  # F,3 long, 0 for no split
    split_edges = edges[splits]  # S sync

    # vertices
    split_vertices = vertices[split_edges].mean(dim=1)  # S,3
    vertices = torch.concat((vertices, split_vertices), dim=0)

    # faces
    side_split = side_vert != 0  # F,3
    # F,3 long, 0 for no split
    shrunk_faces = torch.where(side_split, side_vert, faces)
    new_faces = side_split[:, :, None] * torch.stack((faces, side_vert, shrunk_faces.roll(1, dims=-1)),
                                                     dim=-1)  # F,N=3,C=3
    faces = torch.concat((shrunk_faces, new_faces.reshape(F * 3, 3)))  # 4F,3
    if pack_faces:
        mask = faces[:, 0] != 0
        mask[0] = True
        faces = faces[mask]  # F',3 sync

    return vertices, faces


def pick_edges_to_collapse(
        vertices: torch.Tensor,  # V,3 first unused
        faces: torch.Tensor,  # F,3 long 0 for unused
        edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
        priorities: torch.Tensor,  # E float):
        max_iterations=3,
        stable: bool = False  # only for unit testing
):
    """

    :param vertices:
    :param faces:
    :param edges:
    :param priorities:
    :param stable:
    :return:
    """
    V = vertices.shape[0]
    selected_edges_for_collapse = torch.zeros(
        edges.shape[0], device=vertices.device, dtype=torch.bool)
    for _ in range(max_iterations):
        # check spacing
        _, order = priorities.sort(stable=stable)  # E
        rank = torch.zeros_like(order)
        rank[order] = torch.arange(0, len(rank), device=rank.device)
        vert_rank = torch.zeros(V, dtype=torch.long,
                                device=vertices.device)  # V
        edge_rank = rank  # E
        for i in range(3):
            torch_scatter.scatter_max(src=edge_rank[:, None].expand(-1, 2).reshape(-1), index=edges.reshape(-1), dim=0,
                                      out=vert_rank)
            edge_rank, _ = vert_rank[edges].max(dim=-1)  # E
        candidates_edge_id = (edge_rank == rank).logical_and_(priorities > 0.0)
        selected_edges_for_collapse[candidates_edge_id] = True
        candidates = edges[candidates_edge_id]  # E',2

        # check connectivity
        vert_connections = torch.zeros(
            V, dtype=torch.long, device=vertices.device)  # V
        vert_connections[candidates[:, 0]] = 1  # start
        edge_connections = vert_connections[edges].sum(
            dim=-1)  # E, edge connected to start
        vert_connections.scatter_add_(dim=0, index=edges.reshape(-1),
                                      src=edge_connections[:, None].expand(-1, 2).reshape(-1))  # one edge from start
        vert_connections[candidates] = 0  # clear start and end
        edge_connections = vert_connections[edges].sum(
            dim=-1)  # E, one or two edges from start
        vert_connections.scatter_add_(dim=0, index=edges.reshape(-1),
                                      src=edge_connections[:, None].expand(-1, 2).reshape(
                                          -1))  # one or two edges from start

        # selected_edges = candidates[
        #     vert_connections[candidates[:, 1]] <= 2]
        priorities[candidates_edge_id] = 0
        potential_edge_ids = torch.where(candidates_edge_id == True)[0]
        candidates_edge_id[potential_edge_ids] &= vert_connections[candidates[:, 1]] <= 2
        # selected_edges_for_collapse &=vert_connections[candidates[:, 1]] <= 2
        selected_edges_for_collapse[potential_edge_ids] = candidates_edge_id[potential_edge_ids]
        # find all k ring edges and lower priority to 0
        verts_in_k_rings = torch.zeros(
            V, dtype=torch.long, device=vertices.device)  # V
        edge_visibility = candidates_edge_id.long()
        for _ in range(3):
            torch_scatter.scatter_max(src=edge_visibility[:, None].expand(-1, 2).reshape(-1), index=edges.reshape(-1),
                                      out=verts_in_k_rings)
            edge_visibility = verts_in_k_rings[edges][:,
                                                      0] | verts_in_k_rings[edges][:, 1]
        priorities[edge_visibility.bool()] = 0

    return selected_edges_for_collapse, priorities


def collapse_edges(
        vertices: torch.Tensor,  # V,3 first unused
        faces: torch.Tensor,  # F,3 long 0 for unused
        edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
        priorities: torch.Tensor,  # E float
        threshold,
        collapse_locations: torch.tensor = None,
        max_iterations=5,
        stable: bool = False  # only for unit testing
):  # (vertices,faces)
    selected_edges_for_collapse, _ = pick_edges_to_collapse(
        vertices, faces, edges, priorities.clone(), max_iterations=max_iterations)

    # mask edges for collapses according to priority (ensure at least one edge always gets collapsed), and no very bad collapses happen

    if torch.any(selected_edges_for_collapse):
        normalized_selected_prioriteis = priorities[selected_edges_for_collapse]/torch.max(
            priorities[selected_edges_for_collapse])
        selected_edges_for_collapse[selected_edges_for_collapse.clone(
        )] = normalized_selected_prioriteis > threshold
    collapses = edges[selected_edges_for_collapse]
    # mean vertices
    if collapse_locations is None:
        vertices[collapses[:, 0]] = vertices[collapses].mean(
            dim=1)  # TODO dim?
    else:
        collapse_alpha = collapse_locations[selected_edges_for_collapse].float(
        ).unsqueeze(-1)
        vertices[collapses[:, 0]] = \
            vertices[collapses[:, 0]] * (1 - collapse_alpha) + \
            (collapse_alpha) * vertices[collapses[:, 1]]
        # vertices[:,3:] /=50
    V = vertices.shape[0]
    # update faces
    # for all collapsed edges, face vertices are updated to lower index after collapse.
    # degenerate faces are removed
    dest = torch.arange(0, V, dtype=torch.long, device=vertices.device)  # V
    dest[collapses[:, 1]] = dest[collapses[:, 0]]
    faces = dest[faces]  # F,3 TODO optimize?
    c0, c1, c2 = faces.unbind(dim=-1)
    collapsed = (c0 == c1).logical_or_(c1 == c2).logical_or_(c0 == c2)
    faces[collapsed] = 0

    return vertices, faces, collapsed, selected_edges_for_collapse

def collapse_edges_v2(
        vertices: torch.Tensor,  # V,3 first unused
        faces: torch.Tensor,  # F,3 long 0 for unused
        edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
        priorities: torch.Tensor,  # E float
        threshold,
        collapse_locations: torch.tensor = None,
        max_iterations=5,
        stable: bool = False  # only for unit testing
):
    selected_edges_for_collapse, _ = pick_edges_to_collapse(vertices, faces, edges, priorities.clone(),
                                                            max_iterations=max_iterations)

    # mask edges for collapses according to priority (ensure at least one edge always gets collapsed), and no very bad quality collapses happen

    affected_faces = torch.tensor([], device=vertices.device)
    if torch.any(selected_edges_for_collapse):
        normalized_selected_prioriteis = priorities[selected_edges_for_collapse] / torch.max(
            priorities[selected_edges_for_collapse])
        selected_edges_for_collapse[selected_edges_for_collapse.clone()] = normalized_selected_prioriteis > threshold
        # return affected faces, else no affected faces.
        faces_per_edge, _, _ = calculate_incident_faces_to_edge(faces, edges[selected_edges_for_collapse])
        affected_faces = torch.unique(faces_per_edge.flatten())
    collapses = edges[selected_edges_for_collapse]
    # mean vertices
    if collapse_locations is None:
        vertices[collapses[:, 0]] = vertices[collapses].mean(dim=1)  # TODO dim?
    else:
        collapse_alpha = collapse_locations[selected_edges_for_collapse].float().unsqueeze(-1)
        vertices[collapses[:, 0]] = \
            vertices[collapses[:, 0]] * (1 - collapse_alpha) + collapse_alpha * vertices[collapses[:, 1]]
        # vertices[:,3:] /=50
    V = vertices.shape[0]
    dest = torch.arange(0, V, dtype=torch.long, device=vertices.device)  # V
    dest[collapses[:, 1]] = dest[collapses[:, 0]]
    faces = dest[faces]  # F,3 TODO optimize?
    c0, c1, c2 = faces.unbind(dim=-1)
    collapsed = (c0 == c1).logical_or_(c1 == c2).logical_or_(c0 == c2)  # collapse face ids
    # todo: edge collapse requires dummies?
    faces[collapsed] = 0
    return vertices, faces, collapsed, selected_edges_for_collapse, affected_faces

def get_quality(v, f, max_val=20):
    """
    measure quality by circumradius / 2inradius
    quality = 1 for equilateral triangle
    quality > 1 everything else
    v: vertices (V,3)
    f: faces (F,3)
    max_val: maximum value to clip
    """
    v0, v1, v2 = v[f].unbind(dim=-2)
    a = (v0 - v1).norm(dim=-1)
    b = (v0 - v2).norm(dim=-1)
    c = (v1 - v2).norm(dim=-1)
    s = (a + b + c) / 2
    quality = a * b * c / (8 * (s - a) * (s - b) * (s - c))
    if max_val is not None:
        return quality.clip(max=max_val)
    else:
        return quality


def get_inradius_squared(v, f):
    v0, v1, v2 = v[f].unbind(dim=-2)
    a = (v0 - v1).norm(dim=-1)
    b = (v0 - v2).norm(dim=-1)
    c = (v1 - v2).norm(dim=-1)
    s = (a + b + c) / 2
    return (s-a)*(s-b)*(s-c)/s


def calc_face_collapses(
        vertices: torch.Tensor,  # V,3 first unused
        faces: torch.Tensor,  # F,3 long, 0 for unused
        edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
        face_to_edge: torch.Tensor,  # F,3 long 0 for unused
        face_normals: torch.Tensor,  # F,3
        vertex_normals: torch.Tensor,  # V,3 first unused
        edge_length: torch.Tensor = None,  # E
        min_edge_length: torch.Tensor = None,  # V
        area_ratio=0.5,  # collapse if area < min_edge_length**2 * area_ratio
        shortest_probability=1,
        collapse_low_quality=False
) -> torch.Tensor:  # E edges to collapse

    E = edges.shape[0]
    F = faces.shape[0]

    # face flips
    ref_normals = calc_face_ref_normals(faces, vertex_normals, normalize=False)  # F,3
    face_collapses = (face_normals * ref_normals).sum(dim=-1) < 0  # F

    edge_collapse_ids = torch.unique(face_to_edge[face_collapses])
    # return edge_collapses
    return edge_collapse_ids, face_collapses


def rowwise_in(a, b):
    """
    a - tensor of size a0,c
    b - tensor of size b0,c
    returns - tensor of size a1 with 1 for each row of a in b, 0 otherwise
    """

    # dimensions
    shape1 = a.shape[0]
    shape2 = b.shape[0]
    c = a.shape[1]
    assert c == b.shape[1], "Tensors must have same number of columns"

    a_expand = a.unsqueeze(1).expand(-1, shape2, c)
    b_expand = b.unsqueeze(0).expand(shape1, -1, c)
    # element-wise equality
    return (a_expand == b_expand).all(-1).any(-1)


def select_faces_and_edges(v, f, face_scores, edge_scores, zero_edge_score_threshold=1e-5, num_allowed_faces=None, max_faces_to_split= 4000):
    """
    selects non interfering faces and edges to collapse, such that the overall score increases
    :param v:
    :param f:
    :param face_scores:
    :param edge_scores:
    :return:
    """
    # takes in to account the amount of allowed faces, and make edge collapses and face splits
    #  accordingly
    edges, fe = calc_edges(f, with_dummies=False)
    maximal_face_values_per_edge = get_maximal_face_value_over_edges(edges, f, face_scores)
    # select non interfering edges
    selected_edges_for_collapse, _ = pick_edges_to_collapse(v, f, edges,
                                                            priorities=torch.exp(-edge_scores.clone()),
                                                            max_iterations=5)
    # sort face scores according to descending order
    sorted_face_scores, face_idx = torch.sort(face_scores, descending=True)
    # every edge gets the maximum of its score and the face scores it is incident on. assume no border edges.
    updated_edge_score = \
        torch.max(maximal_face_values_per_edge[selected_edges_for_collapse], edge_scores[selected_edges_for_collapse])
    # sort non-conflicting edge scores according to descending order
    sorted_updated_edge_score, edge_idx = torch.sort(updated_edge_score)
    face_selections = sorted_face_scores[:sorted_updated_edge_score.shape[0]] > sorted_updated_edge_score
    edge_selections = face_selections.clone()
    cleaned_face_idx = face_idx[sorted_face_scores>0]
    selected_faces = cleaned_face_idx[:max_faces_to_split]
    # select edges whose collapse will improve the overall score.
    # note that the selection tensor is of the form: True True .. True False .. False
    if num_allowed_faces is not None:
        face_delta = (num_allowed_faces - f.shape[0]) // 2
        end_of_selection_id = torch.sum(face_selections)
        if face_delta > 1:  # completely under face budget, no need to collapse edges
            edge_selections[:] = False
            amount_to_split = min(max_faces_to_split, face_delta)
            face_selections[:] = False
            cleaned_face_idx = face_idx[sorted_face_scores>0]
            selected_faces = cleaned_face_idx[:amount_to_split]
        elif face_delta <= 1:  # if splitting all faces will go over face budget
            # do not split more faces than allowed
            face_selections[end_of_selection_id + face_delta:end_of_selection_id] = False
            selected_faces = face_idx[:sorted_updated_edge_score.shape[0]][face_selections]
    below_threshold_edges = sorted_updated_edge_score < zero_edge_score_threshold
    # todo: face threshold needs to be above edge threshold or else indexing problems may arise
    selected_edges = torch.where(selected_edges_for_collapse)[0][edge_idx[edge_selections | below_threshold_edges]]
    topology_loss = 0
    return selected_faces, selected_edges, topology_loss


def collapse_given_edges(
        vertices: torch.Tensor,  # V,3 first unused
        faces: torch.Tensor,  # F,3 long 0 for unused
        edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
        selected_edges_for_collapse,
        collapse_locations,
        stable: bool = False  # only for unit testing
):
    # mask edges for collapses according to priority (ensure at least one edge always gets collapsed), and no very bad quality collapses happen

    affected_faces = torch.tensor([], device=vertices.device, dtype=torch.long)
    if torch.any(selected_edges_for_collapse):
        # return affected faces, else no affected faces.
        faces_per_edge, _, _ = calculate_incident_faces_to_edge(faces, edges[selected_edges_for_collapse])
        affected_faces = torch.unique(faces_per_edge.flatten())
    collapses = edges[selected_edges_for_collapse]
    # mean vertices
    if collapse_locations is None:
        vertices[collapses[:, 0]] = vertices[collapses].mean(dim=1)  # TODO dim?
    else:
        collapse_alpha = collapse_locations[selected_edges_for_collapse].float().unsqueeze(-1)
        vertices[collapses[:, 0]] = \
            vertices[collapses[:, 0]] * (1 - collapse_alpha) + collapse_alpha * vertices[collapses[:, 1]]
        # vertices[:,3:] /=50
    V = vertices.shape[0]
    dest = torch.arange(0, V, dtype=torch.long, device=vertices.device)  # V
    dest[collapses[:, 1]] = dest[collapses[:, 0]]
    faces = dest[faces]  # F,3 TODO optimize?
    c0, c1, c2 = faces.unbind(dim=-1)
    collapsed = (c0 == c1).logical_or_(c1 == c2).logical_or_(c0 == c2)  # collapse face ids
    # todo: edge collapse requires dummies?
    faces[collapsed] = 0

    # todo: collapsed faces
    return vertices, faces, collapsed, affected_faces

def calculate_k_hop_average(v, f, k):
    edges_tensor, face_to_edge = calc_edges(f, with_edge_to_face=False)
    adj_mat = calculate_adj_matrix(f)
    ve_mat = calculate_ve_mat(edges_tensor)
    k_power_adj_mat = adj_mat
    for i in range(k - 2):
        k_power_adj_mat = torch.sparse.mm(k_power_adj_mat, adj_mat)
    vertex_matrix = torch.sparse.mm(k_power_adj_mat, ve_mat).coalesce()
    vals = torch.ones(vertex_matrix.values().shape[0], device=v.device)
    vertex_matrix = torch.sparse_coo_tensor(
        vertex_matrix.indices(), vals, device=v.device)
    L = calc_edge_length(v, edges_tensor).unsqueeze(-1)
    L_sum = torch.sparse.mm(vertex_matrix, L)
    edges_num = torch.sparse.mm(
        vertex_matrix, torch.ones_like(L, device=v.device))
    L_v_gpu = L_sum / edges_num

    return L_v_gpu
    # multiply by V to get the sum, multiply by 1's vector to get row num


def calculate_q(v, f, L):
    L_b = torch.mean(L[f], dim=-1)
    b = torch.mean(v[f], dim=-2)
    face_areas = calculate_face_areas(v, f)
    nominator = calculate_vertex_incident_vector(
        v.shape[0], f, face_areas * L_b.unsqueeze(-1) * b)
    denominator = calculate_vertex_incident_vector(
        v.shape[0], f, face_areas * L_b.unsqueeze(-1))
    return nominator / denominator


def tangential_relxation(v, f, L, weights=None):
    device = v.device
    if weights is None:
        step_size_3 = torch.ones(v.shape[0] * 3, device=device)
        step_size_9 = torch.ones(v.shape[0] * 9, device=device)
    else:
        step_size_3 = torch.ones(
            v.shape[0] * 3, device=device) * weights.repeat_interleave(3)
        step_size_9 = torch.ones(
            v.shape[0] * 9, device=device) * weights.repeat_interleave(9)

    vertex_normals = calculate_vertex_normals(v, f)
    nn = torch.bmm(vertex_normals.unsqueeze(-1),
                   vertex_normals.unsqueeze(-2)).flatten()
    # create nn_matrix
    row_ids = torch.arange(3 * v.shape[0], device=device).repeat_interleave(3)
    col_ids = torch.tile(torch.tensor(
        [0, 1, 2], device=device), [3 * v.shape[0]])
    col_ids += torch.arange(v.shape[0], device=device).repeat_interleave(9) * 3
    indices = torch.cat([row_ids.unsqueeze(0), col_ids.unsqueeze(0)], dim=0)
    nn_matrix = torch.sparse_coo_tensor(
        indices=indices, values=nn * step_size_9, device=device)
    identity_matrix_cols = torch.arange(v.shape[0] * 3, device=device)
    identity_matrix_rows = torch.arange(v.shape[0] * 3, device=device)
    identity_matrix_vals = torch.ones_like(identity_matrix_rows, device=device)
    identity_matrix_indices = torch.cat([identity_matrix_rows.unsqueeze(0), identity_matrix_cols.unsqueeze(0)],
                                        dim=0)
    identity_matrix = torch.sparse_coo_tensor(indices=identity_matrix_indices,
                                              values=identity_matrix_vals * step_size_3,
                                              device=device)
    NN = identity_matrix - nn_matrix
    v_unrolled = v.flatten().unsqueeze(-1)
    q = calculate_q(v, f, L).flatten().unsqueeze(-1)
    p = v_unrolled - torch.sparse.mm(NN, v_unrolled - q)
    return p.reshape([-1, 3])
