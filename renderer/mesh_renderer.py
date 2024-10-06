from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *
from torchvision.transforms import GaussianBlur
from ..ROAR.util.func import save_images, load_images

def flip(f):
    tmp = f[:,0]
    f[:,0] = f[:,1]
    f[:,1] = tmp
    return f.contiguous()

@threestudio.register("magicclay-nvdiff-rasterizer")
class NVDiffRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    # expects camera paramters in opengl format
    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        light_positions: Float[Tensor, "B 3"] = None,
        render_normal: bool = True,
        render_rgb: bool = True,
        render_depth: bool = True,
        color_net = None,
        bg_net = None,
        material = None,
        use_supersampled = False,
        shading = None,
        rays_d = None,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]

        if hasattr(self.geometry, 'get_mesh'):
            mesh = self.geometry.get_mesh()
        elif hasattr(self.geometry, 'isosurface'):
            mesh = self.geometry.isosurface()
        else:
            mesh = self.geometry
            if 'allowed_vertices' in mesh.extras:
                mesh.allowed_vertices = mesh.extras['allowed_vertices']
            else:
                mesh.allowed_vertices = None

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        out = {"opacity": mask_aa, "mesh": mesh}

        # returned sampled vertices
        hit = torch.norm(rast,dim=-1)>0
        fv = mesh.v_pos[mesh.t_pos_idx[rast[:,:,:,-1].long()-1]]
        fv0 = fv[:,:,:,0]
        fv1 = fv[:,:,:,1]
        fv2 = fv[:,:,:,2]
        u = rast[:,:,:,[0]]
        v = rast[:,:,:,[1]]
        x = (fv0)*u + (fv1)*v + fv2*(1-u-v)
        # get face id for each pixel
        face_ids = rast[:,:,:,-1].long() - 1
        out.update({"x": x, "hit": hit, "face_ids": face_ids})
        
        # sample color from appearance network if faces are allowed
        if render_normal:
            pertrubations = 0
            if color_net is not None and self.geometry.cfg.optimize_normal_map:
                pertrubations = torch.randn_like(mesh.v_nrm, device=mesh.v_nrm.device)
            
            gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm + pertrubations  , rast, mesh.t_pos_idx)
            gb_normal = F.normalize(gb_normal, dim=-1)
            gb_normal_aa = torch.lerp(
                torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
            )
            gb_normal_aa = self.ctx.antialias(
                gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
            )
            out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

        if render_depth:
            # each pixel is colored according to its distance on the forward camera axis (camera dir). according to depth_to_normal in MVEdit's nerf_utils, depth is inverse 1/z.
            v_forward = torch.tensor([0,0,1], device = 'cuda', dtype=torch.float).reshape([1,-1,1]).expand([c2w.shape[0],-1,-1])
            cams_forward_dir = torch.bmm(-c2w[:,:3,:3],v_forward).transpose(dim0=-1, dim1=-2).unsqueeze(-2)
            inv_z = torch.sum(((x-camera_positions.reshape([c2w.shape[0],1,1,-1]))*cams_forward_dir),dim=-1, keepdim=True)**(-1)
            inv_z[~hit] = 0
            out.update({"depth": inv_z.clip(0,1)})  # in [0, 1]

        if render_rgb:

            if use_supersampled:
                mesh = self.geometry.get_supersampled_mesh(orient_normals_strategy = 'smooth', offset=1e-3)
                v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
                    mesh.v_pos, mvp_mtx
                )
                rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
                mask = rast[..., 3:] > 0
                mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)
                fv = mesh.v_pos[mesh.t_pos_idx[rast[:,:,:,-1].long()-1]]
                fv0 = fv[:,:,:,0]
                fv1 = fv[:,:,:,1]
                fv2 = fv[:,:,:,2]
                u = rast[:,:,:,[0]]
                v = rast[:,:,:,[1]]
                x = (fv0)*u + (fv1)*v + fv2*(1-u-v)
                # get face id for each pixel
                face_ids = rast[:,:,:,-1].long() - 1
       
            selector = mask[..., 0]
            
            features = None
            geo_out = None

            if hasattr(self.geometry, 'forward_color') and self.geometry.forward_color is not None: # geometry is DynamicMesh 
                features = self.geometry.forward_color(mesh.v_pos).detach()
  
                if color_net is not None and features is not None:

                    features[mesh.allowed_vertices] = color_net(mesh.v_pos[mesh.allowed_vertices]) 
                elif color_net is not None:
                    features = color_net(mesh.v_pos) 

            elif hasattr(self.geometry, 'forward_color') and self.geometry.forward_color is None and color_net is not None:
                features = color_net(mesh.v_pos) 
            elif hasattr(self.geometry, 'has_texture') and self.geometry.has_texture is True : # geometry is a Mesh with texture
                bc = torch.cat([u,v, 1-u-v], dim=-1)
                geo_out = self.geometry.get_color(face_ids.flatten(), bc.reshape(-1,3)).detach().reshape_as(bc)


            if features is not None:
                colors = features[:,:3]            
            else:
                colors = torch.ones_like(mesh.v_pos, device=self.device, dtype=torch.float)*0.5

            if geo_out is None:
                geo_out, _ = self.ctx.interpolate_one(colors.contiguous(), rast , mesh.t_pos_idx)

            gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos.contiguous(), rast, mesh.t_pos_idx)
            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            )
            gb_light_positions = light_positions[:, None, None, :].expand(
                -1, height, width, -1
            )
            positions = gb_pos[selector]

            geo_out = {'features' : geo_out[selector]}
            if material is None:
                material = self.material

            if material is not None:
                rgb_fg1 = material(
                    viewdirs=gb_viewdirs[selector],
                    positions=positions,
                    light_positions=gb_light_positions[selector],
                    shading_normal=gb_normal[selector],
                    shading = shading,
                    **geo_out
                )
            else:
                rgb_fg1 = geo_out['features']
                
            if self.material is not None:
                rgb_fg2 = self.material(
                    viewdirs=gb_viewdirs[selector],
                    positions=positions,
                    light_positions=gb_light_positions[selector],
                    shading_normal=gb_normal[selector],
                    shading = shading,
                    **geo_out
                )
            else:
                rgb_fg2 = geo_out['features']

            if hasattr(self.geometry, 'forward_color') and self.geometry.forward_color is not None:

                # output mask should be rendered according to original mesh
                allowed_vertices_mask = torch.zeros(mesh.v_pos.shape[0], dtype = torch.bool).to(rgb_fg1)
                allowed_vertices_mask[mesh.allowed_vertices] = True
                allowed_faces_mask = torch.any(allowed_vertices_mask[mesh.t_pos_idx],dim=-1)
                render_mask =  ((face_ids >= 0) & (allowed_faces_mask[face_ids])).int().unsqueeze(-1).repeat([1,1,1,3])
                render_mask = render_mask.float()
                render_mask = render_mask.transpose(1,-1)
                gaussianBlur = GaussianBlur(kernel_size=3)
                for _ in range(3):
                    render_mask = gaussianBlur.forward(render_mask)
                render_mask = render_mask.transpose(1,-1)
                out.update({"mask": (render_mask>0)})

                gb_rgb_fg1 = torch.zeros(batch_size, height, width, 3).to(rgb_fg1)
                gb_rgb_fg1[selector] = rgb_fg1
                gb_rgb_fg2 = torch.zeros(batch_size, height, width, 3).to(rgb_fg2)
                gb_rgb_fg2[selector] = rgb_fg2
                gb_rgb_fg = gb_rgb_fg1*(render_mask) + gb_rgb_fg2*(1-render_mask)
            else:
                gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg1)
                gb_rgb_fg[selector] = rgb_fg1
                out.update({"mask": mask_aa})
            if bg_net is not None:
                gb_rgb_bg = bg_net(dirs=rays_d)
            else:
                gb_rgb_bg = self.background(dirs=rays_d)
            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

            out.update({"comp_rgb": gb_rgb_aa})

        return out

