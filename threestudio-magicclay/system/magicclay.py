from dataclasses import dataclass, field

import torch
from pathlib import Path
from torchvision.transforms import GaussianBlur
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from .hybrid_base import HybridSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
#from NGLOD.my_code.camera_utils import Rt_to_loc_dir_model, mvp_to_KRt
import torch.nn.functional as F
from ..ROAR.util.func import save_images, load_images
import os
import imageio
from threestudio.models.geometry.base import contract_to_unisphere
from pathlib import Path
from ..ROAR.mesh_optimizer.geometry_scores import point_projections_using_sdf
from ..ROAR.sampling.random_sampler import sample_points_from_meshes, sample_pertrubed_points_from_mesh
from ..ROAR.util.geometry_utils import calculate_vertex_normals, calc_face_normals
from ..ROAR.util.topology_utils import calc_edges
import pysdf
from tqdm import tqdm
import numpy as np
import pickle
import xatlas
import trimesh
import torch_scatter
from .texture_utils import export_with_texture

# def calc_smoothness_loss(v,f):
#     mesh = Meshes([v],[f])
#     return mesh_laplacian_smoothing(mesh)


def create_index_list(max_num, power = 1.5):
    ids = np.arange(max_num)
    ids_exp = [int(i**power) for i in ids]
    ids_trunc = [i for i in ids_exp if i<max_num]
    return ids_trunc


def calculate_laplacian_of_vertices(vertices,faces):
    edges, _ = calc_edges(faces, with_dummies=False) 
    edge_smooth = vertices[edges] # E,2,S
    neighbor_smooth = torch.zeros_like(vertices)  # V,S - mean of 1-ring vertices
    torch_scatter.scatter_mean(src=edge_smooth.flip(dims=[1]).reshape(edges.shape[0] * 2, -1), index=edges.reshape(edges.shape[0] * 2, 1),
                            dim=0, out=neighbor_smooth)
    laplace = vertices - neighbor_smooth[:, :3]  # laplace vector
    return laplace

# from: https://gist.github.com/patrickmineault/21b8d78f423ac8ea4b006f9ec1a1a1a7
def downsample_2d(X, sz):
    """
    Downsamples a stack of square images.    
    Args:
        X: a stack of images (batch, channels, ny, ny).
        sz: the desired size of images.
        
    Returns:
        The downsampled images, a tensor of shape (batch, channel, sz, sz)
    """
    kernel = torch.tensor([[.25, .5, .25], 
                           [.5, 1, .5], 
                           [.25, .5, .25]], device=X.device).reshape(1, 1, 3, 3)
    kernel = kernel.repeat((X.shape[1], 1, 1, 1))
    while sz < X.shape[-1] / 2:
        # Downsample by a factor 2 with smoothing
        mask = torch.ones(1, *X.shape[1:], device = X.device)
        mask = F.conv2d(mask, kernel, groups=X.shape[1], stride=2, padding=1)
        X = F.conv2d(X, kernel, groups=X.shape[1], stride=2, padding=1)
        # Normalize the edges and corners.
        X = X = X / mask
    return F.interpolate(X, size=sz, mode='bilinear')


@threestudio.register("threestudio-magicclay")
class MagicClay(HybridSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        pass

    cfg1: Config
    cfg2: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.imgs_list = []
        self.l2_loss = torch.nn.MSELoss()
        
        self.guidance = threestudio.find(self.cfg1.guidance_type)(self.cfg1.guidance)
        self.prompt_processor = threestudio.find(self.cfg1.prompt_processor_type)(
            self.cfg1.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
    
        self.guidance2 = threestudio.find(self.cfg2.guidance_type)(self.cfg2.guidance)
        self.prompt_processor2 = threestudio.find(self.cfg2.prompt_processor_type)(
            self.cfg2.prompt_processor
        )
        self.prompt_utils2 = self.prompt_processor2()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        
        mesh_render_out = self.system2.renderer(**batch[1], color_net = self.system1.geometry.forward_color, material=self.system1.material,  use_supersampled = True, bg_net = self.system1.background)
        sdf_mesh_render_out = {}

        if not self.training:
            sdf_render_out = self.system1.renderer(**batch[0], ray_dict=None)
        else:
            sdf_render_out = self.system1.renderer(**batch[0], ray_dict=None)
            sdf_mesh_render_out = self.system1.renderer(**batch[1], hits = mesh_render_out['hit'] , x = mesh_render_out['x'].detach())

        return {**sdf_render_out}, {**mesh_render_out}, {**sdf_mesh_render_out}
    
    def on_train_batch_start(self, batch, batch_idx, unused=0):
        super().do_update_step(self.system1.true_current_epoch, self.system1.true_global_step)
        self.system1.on_train_batch_start(batch, batch_idx, unused)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        super().do_update_step(self.system1.true_current_epoch, self.system1.true_global_step)
        self.system1.on_validation_batch_start(batch, batch_idx, dataloader_idx)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        super().do_update_step(self.system1.true_current_epoch, self.system1.true_global_step)
        self.system1.on_test_batch_start(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch, batch_idx):
        super().do_update_step(self.system1.true_current_epoch, self.system1.true_global_step)
        self.system1.predict_step(batch, batch_idx)

    def on_predict_epoch_end(self) -> None:
        super().do_update_step(self.system1.true_current_epoch, self.system1.true_global_step)
        self.system1.on_predict_epoch_end()

    def on_predict_start(self) -> None:
        super().do_update_step(self.system1.true_current_epoch, self.system1.true_global_step)
        self.system1.on_predict_start()

    def on_predict_end(self) -> None:
        super().do_update_step(self.system1.true_current_epoch, self.system1.true_global_step)
        self.system1.on_predict_end()

    def on_fit_start(self) -> None:
        """
        pytorch lightning hook
        :return:
        """
        self.system1.set_save_dir(self._save_dir)
        super().do_update_step(self.system1.true_current_epoch, self.system1.true_global_step)
        self.system1.on_fit_start()
        os.makedirs(self._save_dir, exist_ok=True)
        self.system2.geometry.mesh_optimizer.rayCaster = self.system1.geometry.forward_sdf
        self.camera_list = []
        self.save_steps = create_index_list(self.trainer.max_steps, power=2)

    def training_step(self, batch, batch_idx):

        out, out2, out3 = self(batch)

        opt, _ = self.optimizers()
        opt.zero_grad()
        self.system2.geometry.zero_grad()
        loss = 0.0

        system2_rgb_downsampled = downsample_2d(out2["comp_rgb"].transpose(-1,1), out["comp_rgb"].shape[1]).transpose(-1,1)
        system2_normal_downsampled = downsample_2d(out2["comp_normal"].transpose(-1,1), out["comp_normal"].shape[1]).transpose(-1,1)
        system2_opacity_downsampled = downsample_2d(out2["opacity"].transpose(-1,1), out["opacity"].shape[1]).transpose(-1,1)

        if self.system1.true_global_step > self.system2.geometry.cfg.initial_fit_steps:
            guidance_out = self.guidance(out["comp_rgb"], self.prompt_utils, **batch[0])            
            for name, value in guidance_out.items():
                self.log(f"train/low_res/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.system1.C(self.cfg1.loss[name.replace("loss_", "lambda_")])


        if self.system1.true_global_step > self.system2.geometry.cfg.initial_fit_steps:
            guidance_out2 = self.guidance2(out3["comp_rgb"], self.prompt_utils2, **batch[1])            
            for name, value in guidance_out2.items():
                self.log(f"train/hi_res/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.system2.C(self.cfg2.loss[name.replace("loss_", "lambda_")])

        mesh_vertices = self.system2.geometry.mesh_optimizer.vertices
        mesh_faces = self.system2.geometry.mesh_optimizer.faces

        sampled_points_on_mesh, face_ids = sample_points_from_meshes(mesh_vertices[None,...], mesh_faces[None,...], 100000)
        sampled_points_on_mesh = sampled_points_on_mesh.squeeze(0)
        frozen_faces = self.system2.geometry.faces_for_freeze_loss()
        sample_mask = frozen_faces[face_ids]
        if self.system2.geometry.cfg.allowed_vertices_path != "": #and self.system1.true_global_step > self.system2.geometry.cfg.initial_fit_steps:
            freeze_loss = torch.sum(self.system1.geometry.forward_sdf(sampled_points_on_mesh[sample_mask])**2).sqrt()
            loss += freeze_loss*self.system2.C(self.cfg2.loss.lambda_freeze)

        if self.system1.C(self.cfg1.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.system1.C(self.cfg1.loss.lambda_orient)

        if self.system1.C(self.cfg1.loss.lambda_sparsity) > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.system1.C(self.cfg1.loss.lambda_sparsity)

        if self.system1.C(self.cfg1.loss.lambda_opaque) > 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.system1.C(self.cfg1.loss.lambda_opaque)

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if self.system1.C(self.cfg1.loss.lambda_z_variance) > 0:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.system1.C(self.cfg1.loss.lambda_z_variance)

        #if self.system1.true_global_step > self.system2.geometry.cfg.initial_fit_steps: 
        loss_eikonal = (
            (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2).mean()
        self.log("train/loss_eikonal", loss_eikonal)
        loss += loss_eikonal * self.system1.C(self.cfg1.loss.lambda_eikonal)

        laplace = calculate_laplacian_of_vertices(mesh_vertices, mesh_faces)
        if self.system2.geometry.mesh_optimizer.allowed_vertices is not None:
            smoothness_loss = torch.norm(laplace[self.system2.geometry.mesh_optimizer.allowed_vertices],dim=-1).mean()
        else:
            smoothness_loss = torch.norm(laplace,dim=-1).mean()        

        ##### hybrid losses #####
        hybrid_rgb_loss = self.l2_loss(out2['comp_rgb'].detach()*(1), out3["comp_rgb"]*(1))*self.system1.C(self.cfg1.loss.lambda_rgb)  
        hybrid_rgb_loss += self.l2_loss(system2_rgb_downsampled.detach()*(1), out["comp_rgb"]*(1))*self.system1.C(self.cfg1.loss.lambda_rgb)
        if self.system1.true_global_step > self.system2.geometry.cfg.initial_fit_steps: 
            hybrid_rgb_loss += self.l2_loss(out2['comp_rgb']*(1), out3["comp_rgb"].detach()*(1))*self.system1.C(self.cfg2.loss.lambda_rgb)  
        loss += hybrid_rgb_loss
        self.log("train/loss_hybrid_rgb", hybrid_rgb_loss)    

        hybrid_normal_loss = self.l2_loss(system2_normal_downsampled.detach(), out["comp_normal"])*self.system1.C(self.cfg2.loss.lambda_normal)
        hybrid_normal_loss += self.l2_loss(system2_normal_downsampled, out["comp_normal"].detach())*self.system1.C(self.cfg1.loss.lambda_normal)
        loss += hybrid_normal_loss
        self.log("train/loss_hybrid_normal", hybrid_normal_loss)
        

        opacity_loss = self.l2_loss(system2_opacity_downsampled.detach(), out["opacity"])
        loss += opacity_loss
        self.log("train/loss_hybrid_opacity", opacity_loss)         
        ##### hybrid losses #####

        loss += smoothness_loss*self.system2.C(self.cfg2.loss.lambda_smoothness)
        self.log("train/mesh_smoothness_loss", smoothness_loss)
        for name, value in self.cfg1.loss.items():
            self.log(f"train_params/{name}", self.system1.C(value))
        
        self.manual_backward(loss)
        opt.step()

        if self.system1.true_global_step > self.system2.geometry.cfg.initial_fit_steps:
            self.system2.geometry.step()
            with torch.no_grad():
                self.system2.geometry.remesh(self.system1.true_global_step)
        #return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        if self.system1.exporter.cfg.save_video:
            self.system1.test_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            out1, out2, _ = self(batch)

        save_path = self.save_image_grid(
            f"{self.system1.true_global_step}.png", #-{batch[0]['index'][0]}.png",
            [[
                {
                    "type": "rgb",
                    "img": out1["comp_rgb"][0].detach(),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out1["comp_normal"][0].detach(),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out1
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out1["opacity"][0, :, :, 0].detach(),
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]]
             + [[
                {
                    "type": "rgb",
                    "img": out2["comp_rgb"][0].detach(),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out2["comp_normal"][0].detach(),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out1
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out2["opacity"][0, :, :, 0].detach(),
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]],
        )

        self.imgs_list.append(imageio.imread(save_path))

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        
        # todo: apply slight dilation on texture to avoid seams

        export_with_texture(self.system2.geometry, lambda x: self.system1.material.get_albedo(self.system1.geometry.forward_color(x)),
                             save_dir = self.get_save_path("textured_mesh_{}".format(self.system1.true_global_step)))
        if self.system2.geometry.has_animation:
                np.save( self.get_save_path("animation_weight_matrix_{}.npy".format(self.system1.true_global_step)), self.system2.geometry.get_animation_weight_matrix().cpu().numpy())
        if self.system2.geometry.mesh_optimizer.allowed_vertices is not None:
            np.save( self.get_save_path("allowed_vertices_{}.npy".format(self.system1.true_global_step)), self.system2.geometry.mesh_optimizer.allowed_vertices.cpu().numpy())
        if self.system2.geometry.has_animation:
            np.save( self.get_save_path("animation_weight_matrix_{}.npy".format(self.system1.true_global_step)), self.system2.geometry.get_animation_weight_matrix().cpu().numpy())
        if self.system2.geometry.cfg.allowed_vertices_path != "":
            frozen_faces = self.system2.geometry.faces_for_freeze_loss()
            np.save( self.get_save_path("frozen_faces_{}.npy".format(self.system1.true_global_step)), frozen_faces.cpu().numpy())
        allowed_faces, allowed_edges = self.system2.geometry.mesh_optimizer.get_allowed_faces_and_edges()
        np.save( self.get_save_path("allowed_faces_{}.npy".format(self.system1.true_global_step)), allowed_faces.cpu().numpy())
        # todo: if weight matrix exists, transform it using the vmapping
        
        out1, out2, _ = self(batch)
        
        self.save_image_grid(
            f"it{self.system1.true_global_step}-test/{batch[0]['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out1["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out1["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out1
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out1["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.system1.true_global_step,
        )


    def on_test_epoch_end(self):

        if len(self.imgs_list) > 0:
            imageio.mimsave(self.get_save_path("evolution.mp4"), self.imgs_list, fps=15)


        self.save_img_sequence(
            f"it{self.system1.true_global_step}-test",
            f"it{self.system1.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.system1.true_global_step,
        )