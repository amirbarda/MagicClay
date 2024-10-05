import torch
import nvdiffrast.torch as dr
import xatlas
import numpy as np
from ..ROAR.util.geometry_utils import calculate_vertex_normals

glctx = dr.RasterizeCudaContext()

# for uv extrusion
def supercat(tensors, dim: int = 0):
    """
    Similar to `torch.cat`, but supports broadcasting. For example:

    [M, 32], [N, 1, 64] -- supercat 2 --> [N, M, 96]
    """
    ndim = max(x.ndim for x in tensors)
    tensors = [x.reshape(*[1] * (ndim - x.ndim), *x.shape) for x in tensors]
    shape = [max(x.size(i) for x in tensors) for i in range(ndim)]
    shape[dim] = -1
    tensors = [torch.broadcast_to(x, shape) for x in tensors]
    return torch.cat(tensors, dim)

def gpu_f32(inputs):
    return torch.tensor(inputs, dtype=torch.float32, device=torch.device('cuda:0')).contiguous()

def saturate(x: torch.Tensor):
    return torch.clamp(x, 0.0, 1.0) 

def float4(*tensors):
    tensors = [x if torch.is_tensor(x) else gpu_f32(x) for x in tensors]
    tensors = supercat(tensors, dim=-1)
    assert tensors.shape[-1] == 4
    return tensors

def alpha_blend(background: torch.Tensor, foreground: torch.Tensor):
    alpha = saturate((foreground[:,:,[3]] > 0).float())
    return background * (1 - alpha) + foreground * alpha

def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

@torch.no_grad()
def extrude(x: torch.Tensor, iterations: int = 16):
    kernel = torch.tensor([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])[None, None].to(x)
    for _ in range(iterations):
        extrude = x
        rgb = extrude[:,:,:3]
        alpha = (extrude[:,:,[3]] > 0).float()
        extrude = float4(rgb * alpha, alpha)
        x = extrude
        extrude = extrude.permute(2, 0, 1)[:, None]
        extrude = torch.conv2d(extrude, kernel, padding=1)
        extrude = extrude.squeeze(1).permute(1, 2, 0)
        rgb = extrude[:,:,:3]
        alpha = extrude[:,:,[3]]
        extrude = float4(rgb*(alpha > 0).float() / (alpha + 1e-7), (alpha > 0).float())
        x = alpha_blend(extrude, x)
    return x.clip(0,1)

def render_uv(init_mesh, forward_color, resolution = [512,512]):

    with torch.no_grad():
        mesh_vertices = init_mesh.mesh_optimizer.vertices
        mesh_faces = init_mesh.mesh_optimizer.faces
    # clip space transform 

    vmapping, indices, uvs = xatlas.parametrize(mesh_vertices.detach().cpu().numpy(), mesh_faces.cpu().numpy())
    FT = torch.tensor(indices.astype(np.int32), device = mesh_vertices.device)

    uv_clip = torch.tensor(uvs[None, ...]*2.0 - 1.0, device = mesh_vertices.device)

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # rasterize
    rast, _ = dr.rasterize(glctx, uv_clip4, FT.int(), resolution)

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh_vertices[None, ...], rast, mesh_faces.int())
    
    # TODO: find which face in rast is in allowed faces
    allowed_faces, _ = init_mesh.mesh_optimizer.get_allowed_faces_and_edges(mode='relaxed')
    allowed_faces = torch.nn.functional.pad(allowed_faces,(1,0))
    pixels_in_allowed_faces = allowed_faces[rast.to(dtype=torch.int64)][...,3].view(1, -1)

    sh = gb_pos.shape
    if init_mesh.forward_color is not None:
        shaded_col = init_mesh.forward_color(gb_pos.view(1, -1, 3)).clip(0,1)
        shaded_col[pixels_in_allowed_faces] = forward_color(gb_pos.view(1, -1, 3)[pixels_in_allowed_faces.squeeze(-1)]).clip(0,1)
    else:
        shaded_col = forward_color(gb_pos.view(1, -1, 3)).clip(0,1)

    shaded_col = shaded_col.view(*sh)        

    # fix the texture seams
    shaded_col = extrude(torch.cat((shaded_col[0], rast[0, ...,[3]]), dim=-1), iterations=3)

    return shaded_col.float()


def export_with_texture(init_mesh, forward_color, save_dir = ''):

    from PIL import Image
    import cyobj.io as mio
    import os
    from pathlib import Path

    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    with torch.no_grad():
        mesh_vertices = init_mesh.mesh_optimizer.vertices
        mesh_faces = init_mesh.mesh_optimizer.faces

        # output texture of mesh as .mtl file
        vmapping, indices, uvs = xatlas.parametrize(mesh_vertices.cpu().numpy(), mesh_faces.cpu().numpy())
        VT = mesh_vertices[vmapping.astype(np.int32)]
        VN = calculate_vertex_normals(mesh_vertices, mesh_faces)
        FT = indices.astype(np.int64)
        uvs = torch.tensor(uvs, device = mesh_vertices.device)
        indices = torch.tensor(indices.astype(np.int64), device = mesh_vertices.device, dtype = torch.long)
        im = render_uv(init_mesh, forward_color)
        image = Image.fromarray((im.squeeze(0)*255).cpu().numpy().astype(np.uint8))
        image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
        image.save(os.path.join(save_dir,"texture.png"))
        uvs = uvs[:,:2].cpu().numpy()
        rotation_matrix = torch.tensor([[1,0,0],
                                        [0,0, 1],
                                        [0,-1,0]], device = 'cuda', dtype=torch.float).unsqueeze((0))
        mesh_vertices = torch.bmm(rotation_matrix.expand([mesh_vertices.shape[0],-1,-1]),  mesh_vertices.unsqueeze(-1))
        mesh_vertices = mesh_vertices.squeeze(-1)
        mio.write_obj(os.path.join(save_dir,"final_mesh.obj"), mesh_vertices.detach().cpu().numpy().astype(np.float64), mesh_faces.cpu().numpy().astype(np.int64), uvs.astype(np.float64), FT, None, None, path_img = "texture.png")
