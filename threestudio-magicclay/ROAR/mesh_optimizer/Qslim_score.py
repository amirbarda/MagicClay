from ..util.geometry_utils import calculate_face_normals_and_areas, calculate_vertex_normals, \
    calculate_head_angles_per_vertex_per_face, calculate_edge_normals, calculate_generalize_vertex_normals
from ..util.topology_utils import calc_edges, calculate_vertex_incident_vector, calculate_adjacent_faces, \
    calculate_vertex_incident_scalar, calculate_vertex_incident_vector_on_edges
from ..util.remeshing_utils import remove_dummies, prepend_dummies
from ..util.func import load_mesh, normalize_vertices, save_mesh_properly
import torch
import numpy as np
import torch_scatter
import os
import igl
import torch.nn.functional as tfunc


def calc_for_face_vertices_normals(fv, eps=1e-8):
    fv0 = fv[:, 0]
    fv1 = fv[:, 1]
    fv2 = fv[:, 2]
    face_normals = torch.cross(fv1 - fv0, fv2 - fv0, dim=-1)
    normal_mask = torch.norm(face_normals, dim=-1) < eps
    face_normals[~normal_mask] /= torch.norm(face_normals[~normal_mask], dim=-1, keepdim=True)
    face_normals[normal_mask] = 0
    return face_normals


def calc_head_angle(fv, head_vertex_id):
    selection_ids = torch.arange(fv.shape[0], device=fv.device)
    fv0 = fv[selection_ids, head_vertex_id]
    fv1 = fv[selection_ids, (head_vertex_id + 1) % 3]
    fv2 = fv[selection_ids, (head_vertex_id + 2) % 3]
    mask = (torch.norm(fv1 - fv0, dim=-1) < 1e-9) | (torch.norm(fv2 - fv0, dim=-1) < 1e-9)
    cos = torch.sum(tfunc.normalize(fv1 - fv0) * tfunc.normalize(fv2 - fv0), dim=-1, keepdim=True)
    cos = torch.clip(cos, max=1, min=-1)
    angle = torch.acos(cos)
    #assert torch.all(~torch.isnan(angle))
    angle[mask] = 0
    return angle


def calculate_double_face_areas(fv):
    fv0 = fv[:, 0]
    fv1 = fv[:, 1]
    fv2 = fv[:, 2]
    double_face_areas = torch.norm(torch.cross(fv1 - fv0, fv2 - fv0, dim=-1), dim=-1, keepdim=True)
    return double_face_areas


def calculate_frozen_faces(fv_frozen):
    # for each potential face calculate if is allowed
    fv_frozen0 = fv_frozen[:, 0]
    fv_frozen1 = fv_frozen[:, 1]
    fv_frozen2 = fv_frozen[:, 2]
    faces_frozen = fv_frozen0 & fv_frozen1 & fv_frozen2
    return faces_frozen


def calc_face_vertices_quality(fv):
    # lower is better
    fv0 = fv[:, 0]
    fv1 = fv[:, 1]
    fv2 = fv[:, 2]
    # calculate quality
    a = (fv0 - fv1).norm(dim=-1)
    b = (fv0 - fv2).norm(dim=-1)
    c = (fv1 - fv2).norm(dim=-1)
    s = (a + b + c) / 2
    quality = a * b * c / (8 * (s - a) * (s - b) * (s - c))
    # degenerate (collapsed) faces have quality 1 (best)
    quality = torch.nan_to_num(quality, nan=1)
    return quality



def simulated_faces_after_collapse(v, f, v_edge, quality_threshold=5, with_dummies=True, v_mask = None):
    """

    :param v:
    :param f:
    :param v_edge:
    :param quality_threshold:
    :return:
    """
    adj_faces, num_of_incident_faces, starting_idx = calculate_adjacent_faces(f)
    edges, _ = calc_edges(f, with_edge_to_face=False, with_dummies = with_dummies)
    v_ids_for_incident_faces_per_edge = torch.repeat_interleave(edges.flatten(),
                                                                repeats=num_of_incident_faces[edges.flatten()])
    offset_vector = torch.arange(v_ids_for_incident_faces_per_edge.shape[0], device=v.device)
    cumsum = torch.cumsum(num_of_incident_faces[edges.flatten()], dim=0).roll(1)
    cumsum[0] = 0
    to_subtract = cumsum.repeat_interleave(repeats=num_of_incident_faces[edges.flatten()])
    offset_vector -= to_subtract
    starting_idx_for_v_ids = starting_idx[v_ids_for_incident_faces_per_edge]
    starting_idx_for_v_ids += offset_vector
    # create tensor that contains incident faces per edge
    faces_per_edge = adj_faces[starting_idx_for_v_ids]
    incident_face_vertices_ids_per_edge = f[faces_per_edge]
    incident_face_vertices_per_edge = v[incident_face_vertices_ids_per_edge]
    edge_degrees = torch.sum(num_of_incident_faces[edges], dim=-1) # num_of_incident_faces is per vertex, edge degrees is the sum of its vertex degrees
    # replace vertices to v_edge wherever mask is true
    edge_vertices0 = edges[:, [0]].repeat_interleave(repeats=edge_degrees, dim=0)
    edge_vertices0 = torch.tile(edge_vertices0, [1, 3])
    edge_vertices1 = edges[:, [1]].repeat_interleave(repeats=edge_degrees, dim=0)
    edge_vertices1 = torch.tile(edge_vertices1, [1, 3])
    edge_vertices0_replace_mask = (edge_vertices0 == incident_face_vertices_ids_per_edge)
    if with_dummies:
        edge_vertices0_replace_mask &= (edge_vertices0 != 0)
    edge_vertices1_replace_mask = (edge_vertices1 == incident_face_vertices_ids_per_edge)
    if with_dummies:
        edge_vertices1_replace_mask &= (edge_vertices1 != 0)
    v0_replace = v_edge.repeat_interleave(repeats=edge_degrees, dim=0)[torch.any(edge_vertices0_replace_mask, dim=-1)]
    incident_face_vertices_per_edge[edge_vertices0_replace_mask] = v0_replace
    v1_replace = v_edge.repeat_interleave(repeats=edge_degrees, dim=0)[torch.any(edge_vertices1_replace_mask, dim=-1)]
    incident_face_vertices_per_edge[edge_vertices1_replace_mask] = v1_replace

    # incident_face_vertices_per_edge now contains state after collapse
    normals_before = calc_for_face_vertices_normals(v[incident_face_vertices_ids_per_edge])
    # calc face normals after
    normals_after = calc_for_face_vertices_normals(incident_face_vertices_per_edge)
        

    flip_detection = torch.sum(normals_before * normals_after, dim=-1) < 0
    # scatter max for all vertices to edges
    repeated_edge_ids = torch.arange(edges.shape[0], device=v.device).repeat_interleave(repeats=edge_degrees)
    edge_mask = torch_scatter.scatter_max(flip_detection.float(), repeated_edge_ids, dim=0)[0]

    if quality_threshold is not None:
        incident_faces_quality = calc_face_vertices_quality(incident_face_vertices_per_edge)
        bad_face_detection = incident_faces_quality > quality_threshold
        repeated_edge_ids = torch.arange(edges.shape[0], device=v.device).repeat_interleave(repeats=edge_degrees)
        face_quality_mask = torch_scatter.scatter_max(bad_face_detection.float(), repeated_edge_ids, dim=0)[0]
        edge_mask = edge_mask + face_quality_mask


    if v_mask is not None:
        fv_frozen = v_mask[incident_face_vertices_ids_per_edge]   
        frozen_faces_before = calculate_frozen_faces(fv_frozen)
        # find if edge is border edge, for and repeat for each vertex incidient to edge
        border_edge_mask = v_mask[edges].sum(dim=-1)==1 # #EV x 1, true if edge is border edge, false otherwise
        border_edge_vertices_mask = border_edge_mask.repeat_interleave(repeats=edge_degrees).unsqueeze(-1)
        # replace all border edges with frozen vertices after simulated collapse
        fv_frozen_after = fv_frozen
        fv_frozen_after[edge_vertices0_replace_mask & border_edge_vertices_mask] = True
        fv_frozen_after[edge_vertices1_replace_mask & border_edge_vertices_mask] = True
        # find list of incident faces 
        frozen_faces_after = calculate_frozen_faces(fv_frozen_after)
        # edge collapse should not turn an allowed face to a frozen one, but in case of border edge the two incident triangles are destroyed so they can become frozen
        is_incident_to_edge_to_collapse = (edge_vertices0_replace_mask | edge_vertices1_replace_mask).sum(dim=-1) == 2
        bad_border_detection = (frozen_faces_after != frozen_faces_before)
        bad_border_detection[is_incident_to_edge_to_collapse] = False
        repeated_edge_ids = torch.arange(edges.shape[0], device=v.device).repeat_interleave(repeats=edge_degrees)
        allowed_vertices_mask = torch_scatter.scatter_max(bad_border_detection.float(), repeated_edge_ids, dim=0)[0]
        edge_mask = edge_mask + allowed_vertices_mask

    return edge_mask > 0



def calculate_Q_f(v, f):
    face_normals, _ = calculate_face_normals_and_areas(v, f)
    d = -torch.sum(v[f[:, 0]] * face_normals, dim=-1, keepdim=True)  # OK
    A = torch.bmm(face_normals.unsqueeze(-1), face_normals.unsqueeze(-1).transpose(-1, -2))
    b = d * face_normals
    c = d ** 2
    return A, b, c


def calculate_Q_e(v, f, edges):
    edge_normal = calculate_edge_normals(v, f)
    d = -torch.sum(v[edges[:, 0]] * edge_normal, dim=-1, keepdim=True)  # OK
    A = torch.bmm(edge_normal.unsqueeze(-1), edge_normal.unsqueeze(-1).transpose(-1, -2))
    b = d * edge_normal
    c = d ** 2
    return A, b, c


def calculate_Q_vv(v, f):
    vertex_normals = calculate_generalize_vertex_normals(v, f)
    d = -torch.sum(v * vertex_normals, dim=-1, keepdim=True)  # OK
    A = torch.bmm(vertex_normals.unsqueeze(-1), vertex_normals.unsqueeze(-1).transpose(-1, -2))
    b = d * vertex_normals
    c = d ** 2
    return A, b, c


def calculate_Q_v(v, f, with_dummies):
    edges, _ = calc_edges(f, with_dummies = with_dummies)
    A, b, c = calculate_Q_f(v, f)
    A_e, b_e, c_e = calculate_Q_e(v, f, edges)
    A_vv, b_vv, c_vv = calculate_Q_vv(v, f)
    Abc = torch.cat([A.reshape(f.shape[0], -1), b, c], dim=-1)
    Abc_v = calculate_vertex_incident_vector(v.shape[0], f, Abc*100)
    Abc_e = torch.cat([A_e.reshape(edges.shape[0], -1), b_e, c_e], dim=-1)
    Abc_v += calculate_vertex_incident_vector_on_edges(v.shape[0], edges, Abc_e)
    A_v = Abc_v[:, :9].reshape(-1, 3, 3) + A_vv*100
    b_v = Abc_v[:, 9:12] + b_vv*100
    c_v = Abc_v[:, 12] + c_vv.flatten()*100
    return A_v, b_v, c_v


def Qslim_plane_distance(v_bar, A_tot, b_tot, c_tot):
    A_score = torch.bmm(v_bar.unsqueeze(-2), torch.bmm(A_tot, v_bar.unsqueeze(-1))).flatten()
    b_score = 2 * torch.sum(b_tot * v_bar, dim=-1)
    distance = A_score + b_score + c_tot
    distance[distance < 0] = 0
    return distance


def calculate_QSlim_score(v, f, face_colors=None, color_score = 1, quality_threshold=None, penalty_weight = 10, avoid_folding_faces=True, min_edge_length = 1e-5, with_dummies=False, output_debug_info = False, v_mask = None):
    """
    higher QSlim score = edge encodes more geometry
    :param face_colors:
    :param face_colors:
    :param v:
    :param f:
    :param quality_threshold:
    :param avoid_folding_faces:
    :param vertex_norm_threshold:
    :param with_dummies:
    :return:
    """

    debug_info = {}

    A_v, b_v, c_v = calculate_Q_v(v, f, with_dummies)
    edges, _ = calc_edges(f, with_dummies = with_dummies)
    v0 = edges[:, 0]
    v1 = edges[:, 1]
    v0_bar = v[v0]
    v1_bar = v[v1]
    # return the minimal scores AND the location of collapse
    A_tot = A_v[v0] + A_v[v1]
    b_tot = b_v[v0] + b_v[v1]
    c_tot = c_v[v0] + c_v[v1]
    qslim_score_0 = Qslim_plane_distance(v0_bar, A_tot, b_tot, c_tot) # score of moving v1 to v0
    qslim_score_1 = Qslim_plane_distance(v1_bar, A_tot, b_tot, c_tot) # score of moving v0 to v1
    edge_lengths = torch.norm(v1_bar - v0_bar, dim=-1)
    if avoid_folding_faces or v_mask is not None:
        # multiply qslim score by this mask
        qslim_score_0[simulated_faces_after_collapse(v, f, v0_bar, quality_threshold=quality_threshold,
                                                     with_dummies=with_dummies, v_mask = v_mask) & (edge_lengths>min_edge_length)] = np.inf
        qslim_score_1[simulated_faces_after_collapse(v, f, v1_bar, quality_threshold=quality_threshold,
                                                     with_dummies=with_dummies, v_mask = v_mask) & (edge_lengths>min_edge_length)] = np.inf
        

    if v_mask is not None:
        qslim_score_1[v_mask[v0]] = np.inf
        qslim_score_0[v_mask[v1]] = np.inf

    qslim_scores = torch.stack([qslim_score_0, qslim_score_1])
    min_qslim_val, min_qslim_id = torch.min(qslim_scores, dim=0)

    if output_debug_info:
        debug_info['q0_arr'] = qslim_score_0.cpu().numpy()
        debug_info['q1_arr'] = qslim_score_1.cpu().numpy()

    return min_qslim_val, min_qslim_id.float(), debug_info

