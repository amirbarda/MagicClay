import torch
from torch_scatter import scatter_max
from ..util.geometry_utils import calculate_face_folds, calculate_vertex_normals, calc_face_normals
from ..util.topology_utils import calculate_vertex_incident_scalar, get_maximal_face_value_over_vertices, calc_edges
from ..util.remeshing_utils import calc_face_collapses
import numpy as np
import igl


def gradient(x, f, method='autodiff'):
    """Compute gradient.
    """
    if method == 'autodiff':
        with torch.enable_grad():
            x = x.requires_grad_(True)
            y = f(x)
            grad = torch.autograd.grad(y, x, 
                                       grad_outputs=torch.ones_like(y), create_graph=True)[0]
    elif method == 'tetrahedron':
        h = 1.0 / (64.0 * 3.0)
        k0 = torch.tensor([ 1.0, -1.0, -1.0], device=x.device, requires_grad=False)
        k1 = torch.tensor([-1.0, -1.0,  1.0], device=x.device, requires_grad=False)
        k2 = torch.tensor([-1.0,  1.0, -1.0], device=x.device, requires_grad=False)
        k3 = torch.tensor([ 1.0,  1.0,  1.0], device=x.device, requires_grad=False)
        h0 = torch.tensor([ h, -h, -h], device=x.device, requires_grad=False)
        h1 = torch.tensor([-h, -h,  h], device=x.device, requires_grad=False)
        h2 = torch.tensor([-h,  h, -h], device=x.device, requires_grad=False)
        h3 = torch.tensor([ h,  h,  h], device=x.device, requires_grad=False)
        h0 = x + h0
        h1 = x + h1
        h2 = x + h2
        h3 = x + h3
        h0 = h0.detach()
        h1 = h1.detach()
        h2 = h2.detach()
        h3 = h3.detach()
        h0 = k0 * f(h0)
        h1 = k1 * f(h1)
        h2 = k2 * f(h2)
        h3 = k3 * f(h3)
        grad = (h0+h1+h2+h3) / (h*4.0)
    elif method == 'finitediff':
        min_dist = 1.0/(64.0 * 3.0)
        eps_x = torch.tensor([min_dist, 0.0, 0.0], device=x.device)
        eps_y = torch.tensor([0.0, min_dist, 0.0], device=x.device)
        eps_z = torch.tensor([0.0, 0.0, min_dist], device=x.device)

        grad = torch.cat([f(x + eps_x) - f(x - eps_x),
                          f(x + eps_y) - f(x - eps_y),
                          f(x + eps_z) - f(x - eps_z)], dim=-1)
        grad = grad / (min_dist*2.0)
    elif method == 'multilayer':
        grad = []
        with torch.enable_grad():
            _y = f.sdf(x, return_lst=True)
            for i in range(len(_y)):
                _grad = torch.autograd.grad(_y[i], x, 
                                           grad_outputs=torch.ones_like(_y[i]), create_graph=True)[0]
                grad.append(_grad)
        return grad
    else:
        raise NotImplementedError

    return grad

def max_distance_score(projected_points, faces, reg_sampler):
    face_ids = reg_sampler.sampled_face_ids
    dists = torch.norm(projected_points - reg_sampler.sampled_vertices, dim=-1)
    out, argmax = scatter_max(dists, face_ids)
    face_scores = torch.zeros(faces.shape[0], device=projected_points.device)
    face_scores[argmax != 50000] = dists[argmax[argmax != 50000]]
    return face_scores


def random_face_score(faces):
    face_scores = torch.rand(faces.shape[0], device = faces.device)
    return face_scores

def calculate_double_face_areas(fv):
    fv0 = fv[:, 0]
    fv1 = fv[:, 1]
    fv2 = fv[:, 2]
    double_face_areas = torch.norm(torch.cross(fv1 - fv0, fv2 - fv0, dim=-1), dim=-1, keepdim=True)
    return double_face_areas

def calc_for_face_vertices_areas(fv, eps=1e-8):
    fv0 = fv[:, 0]
    fv1 = fv[:, 1]
    fv2 = fv[:, 2]
    face_normals = torch.cross(fv1 - fv0, fv2 - fv0, dim=-1)
    normal_mask = torch.norm(face_normals, dim=-1) < eps
    face_normals[~normal_mask] /= torch.norm(face_normals[~normal_mask], dim=-1, keepdim=True)
    face_normals[normal_mask] = 0
    return face_normals

def calc_for_face_vertices_normals(fv, eps=1e-8):
    fv0 = fv[:, 0]
    fv1 = fv[:, 1]
    fv2 = fv[:, 2]
    face_normals = torch.cross(fv1 - fv0, fv2 - fv0, dim=-1)
    normal_mask = torch.norm(face_normals, dim=-1) < eps
    face_normals[~normal_mask] /= torch.norm(face_normals[~normal_mask], dim=-1, keepdim=True)
    face_normals[normal_mask] = 0
    return face_normals

def face_scores_face_folds(projected_points, faces, reg_sampler, face_normals, projected_points_mask=None, curvature_threshold=0.02, mode = 'face split'):
    device = projected_points.device

    sampled_faces = reg_sampler.sampled_faces
    face_ids = reg_sampler.sampled_face_ids
    if projected_points_mask is None:
        sampled_face_mask = None
    else:
        sampled_face_mask = ~(torch.any(projected_points_mask[sampled_faces],dim=-1))
        sampled_face_mask = sampled_face_mask.bool()
    updated_vertex_normals = calculate_vertex_normals(projected_points, sampled_faces[sampled_face_mask])
    if projected_points_mask is not None:
        updated_vertex_normals[projected_points_mask] = 0
    fold_per_sampled_face_after = calculate_face_folds(projected_points, sampled_faces,
                                                       vertex_normals=updated_vertex_normals)
    fold_per_sampled_face_after = torch.nan_to_num(fold_per_sampled_face_after, nan=0)

    fold_per_sampled_face_after[fold_per_sampled_face_after > 1] = 0
    fold_per_sampled_face_after[~sampled_face_mask] = 0
    # check if normals are flipped
    face_normals_before = face_normals[face_ids]
    face_normals_after = calc_for_face_vertices_normals(projected_points[reg_sampler.sampled_faces])
    flipped_normals = torch.sum(face_normals_before * face_normals_after, dim=-1) < 0
    fold_per_sampled_face_after[flipped_normals] = 0

    fold_per_sampled_face_after[fold_per_sampled_face_after < curvature_threshold] = 0
    num_of_non_zero_per_face = torch.zeros(faces.shape[0], device=device, dtype=torch.float)
    face_scores_after = torch.zeros(faces.shape[0], device=device, dtype=torch.float)
    fold_per_sampled_face_after = torch.nan_to_num(fold_per_sampled_face_after, nan=0)
    face_scores_after = torch.index_add(face_scores_after, dim=0, index=face_ids,
                                        source=fold_per_sampled_face_after)
    

    edge_flip_correction_per_face = torch.zeros_like(fold_per_sampled_face_after, device = fold_per_sampled_face_after.device)
    edge_flip_correction_per_face[~sampled_face_mask] = 1
    edge_flip_correction = torch.zeros(faces.shape[0], device=device, dtype=torch.float)
    edge_flip_correction = torch.index_add(edge_flip_correction, dim=0, index=face_ids,
                                        source=edge_flip_correction_per_face)
    edge_flip_correction /= reg_sampler.amount_of_vertices_for_each_face

    return face_scores_after, edge_flip_correction, fold_per_sampled_face_after,face_normals_after, updated_vertex_normals, sampled_face_mask, edge_flip_correction_per_face


@torch.no_grad()
def point_projections_using_sdf(points, sdf):
    grads = gradient(points, sdf, method='finitediff')
    dists = sdf(points)
    projection = -dists * grads / (torch.norm(grads, dim=-1, keepdim=True)+ 1e-5)
    return projection

@torch.no_grad()
def face_split_sdf_sphere_tracing(all_vertices, vertex_normals, sdf, iterations = 5, return_debug_info = False):
    with torch.no_grad():
        projected_points = all_vertices.clone()
        for _ in range(iterations):
            projection = point_projections_using_sdf(projected_points, sdf)
            projection = vertex_normals*torch.sum(projection * vertex_normals, dim=-1, keepdim=True)
            projected_points = projection + projected_points
    projected_sdf_vals = sdf(projected_points)
    original_sdf_vals = sdf(all_vertices)
    projection_mask = torch.abs(projected_sdf_vals.flatten())>torch.abs(original_sdf_vals.flatten())
    projection_mask |= torch.abs(projected_sdf_vals.flatten())>1e-6
    return projected_points, projection_mask, original_sdf_vals, projected_sdf_vals