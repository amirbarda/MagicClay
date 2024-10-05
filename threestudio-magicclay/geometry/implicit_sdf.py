import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.misc import broadcast, get_rank
from threestudio.utils.typing import *
from ..ROAR.sampling.random_sampler import sample_pertrubed_points_from_mesh, sample_points_from_meshes
from ..ROAR.util.func import normalize_vertices, read_mtlfile
#from NGLOD.lib.diffutils import gradient
import igl
import imageio as iio
from PIL import Image
from pathlib import Path
#from wavefront_reader import read_wavefront, read_mtlfile, read_objfile

@threestudio.register("magicclay-implicit-sdf")
class ImplicitSDF(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        initial_sdf_weights: Optional[str] = None
        n_input_dims: int = 3
        n_feature_dims: int = 3
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: Union[
            float, str
        ] = 0.01  # in [float, "progressive"]
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"
        force_shape_init: bool = False
        sdf_bias_mtl: str = ""
        sdf_bias: Union[float, str] = 0.0
        sdf_bias_params: Optional[Any] = None
        initial_steps : int = 1000
        save_initial_sdf: bool = False
        # no need to removal outlier for SDF
        isosurface_remove_outliers: bool = False

    cfg: Config
    
    def initialize_using_mesh(self):

        mesh_path = self.cfg.sdf_bias[5:]
        if not os.path.exists(mesh_path):
            raise ValueError(f"Mesh file {mesh_path} does not exist.")
        
        v,tc,_,f,ftc,_ = igl.read_obj(mesh_path)

        mesh_vertices = torch.tensor(v, device= self.device)
        mesh_vertices = mesh_vertices.unsqueeze(0)
        mesh_faces = torch.tensor(f, device = self.device).unsqueeze(0)

        from pysdf import SDF

        print('initializing sdf from base mesh, {} steps'.format(self.cfg.initial_steps))

        sdf = SDF(v, f)

        def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
            # add a negative signed here
            # as in pysdf the inside of the shape has positive signed distance
            return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                points_rand
            )[..., None]
            

        get_gt_sdf = func
        self.bbox = self.bbox.to(self.device)
        #self.bbox = torch.stack([mesh_vertices[0].min(dim=0)[0], mesh_vertices[0].max(dim=0)[0]], dim=0)
        num_samples = 1000000
    
        # if mtl file exists, calculate colors for points on the surface 
        # Initialize SDF to a given shape when no weights are provided or force_shape_init is True

        if self.cfg.sdf_bias_mtl != "" and Path(self.cfg.sdf_bias_mtl).exists() :

            _,tc,_,_,ftc,_ = igl.read_obj(mesh_path)
            tc = torch.tensor(tc, device=self.device, dtype = torch.float)
            ftc = torch.tensor(ftc, device=self.device, dtype = torch.long)
            material = read_mtlfile(self.cfg.sdf_bias_mtl)
            parent_path = Path(self.cfg.sdf_bias_mtl).parent
            texture_image = iio.imread(Path(parent_path,material[next(iter(material))]['map_Kd']))[:,:,:3]
            image = Image.fromarray(texture_image)
            image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
            texture_image = torch.tensor(np.array(image), device=self.device)
            tc = tc.clip(0,1)
        optim = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        from tqdm import tqdm
        for i in (pbar:=tqdm(
            range(self.cfg.initial_steps),
            desc=f"Initializing SDF to a(n) {self.cfg.sdf_bias}:",
            disable=get_rank() != 0,
        )):
            points_rand = (torch.rand((num_samples//8, 3), dtype=torch.float32).to(self.device)-0.5)*(self.bbox[1,:] - self.bbox[0,:])
            pertrubed_points_on_surface = sample_pertrubed_points_from_mesh(mesh_vertices, mesh_faces, num_samples*3//8).squeeze(0)
            points_on_surface, bc, face_ids = sample_points_from_meshes(mesh_vertices, mesh_faces, num_samples//2, return_bc_coords = True)
            points_on_surface = points_on_surface.squeeze(0)
            points = torch.cat([points_rand, points_on_surface, pertrubed_points_on_surface])
            sdf_gt = -sdf(points.detach().cpu().numpy())
            sdf_gt = torch.tensor(sdf_gt, device=self.device, dtype=torch.float).unsqueeze(-1)
            sdf_pred = self.forward_sdf_bias(points)
            loss = F.mse_loss(sdf_pred, sdf_gt)
                        
            # get barycentric coordinates of points
            if self.cfg.sdf_bias_mtl != "" and Path(self.cfg.sdf_bias_mtl).exists() :
                # get texture coordinate of each point
                points_tc = torch.sum(tc[ftc[face_ids]]*bc[...,None], dim=1).clip(0)
                points_tc[:,0] *= texture_image.shape[0]-1
                points_tc[:,1] *= texture_image.shape[1]-1
                points_tc = points_tc.round().to(torch.long)
                color_gt = texture_image[points_tc[:,1], points_tc[:,0]]/255
                color_pred = self.forward_color(points_on_surface)
                loss += F.mse_loss(color_pred, color_gt)

            pbar.set_postfix_str('loss: {:.4E}'.format(loss.item()))
            optim.zero_grad()
            loss.backward()
            optim.step()
  
        # explicit broadcast to ensure param consistency across ranks
        for param in self.parameters():
            broadcast(param, src=0)

    def configure(self) -> None:
        super().configure()
        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        ).to(self.device)
        self.sdf_network = get_mlp(
            self.encoding.n_output_dims, 1, self.cfg.mlp_network_config
        ).to(self.device)
        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            ).to(self.device)
        
        if self.cfg.normal_type == "pred":
            self.normal_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            ).to(self.device)
        if self.cfg.isosurface_deformable_grid:
            assert (
                self.cfg.isosurface_method == "mt"
            ), "isosurface_deformable_grid only works with mt"
            self.deformation_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            ).to(self.device)
        
        self.bbox = self.bbox.to(self.device)
        # if sdf_bias is mesh: initialize mesh network, unless weight path is provided
        if not isinstance(self.cfg.sdf_bias,float) and self.cfg.sdf_bias.startswith('mesh'):
            self.encoding_bias = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        ).to(self.device)
            self.sdf_bias_network = get_mlp(
            self.encoding.n_output_dims, 1, self.cfg.mlp_network_config
        ).to(self.device)
            
            if self.cfg.initial_sdf_weights is not None:
                self.load_state_dict(torch.load(self.cfg.initial_sdf_weights))
            #if not self._resumed:
            self.initialize_using_mesh()


        self.finite_difference_normal_eps: Optional[float] = None

    def initialize_shape(self) -> None:
        if self.cfg.shape_init is None and not self.cfg.force_shape_init:
            return

        # do not initialize shape if weights are provided
        if self.cfg.weights is not None and not self.cfg.force_shape_init:
            return

        if self.cfg.sdf_bias != 0.0:
            threestudio.warn(
                "shape_init and sdf_bias are both specified, which may lead to unexpected results."
            )

        get_gt_sdf: Callable[[Float[Tensor, "N 3"]], Float[Tensor, "N 1"]]
        assert isinstance(self.cfg.shape_init, str)
        if self.cfg.shape_init == "ellipsoid":
            assert (
                isinstance(self.cfg.shape_init_params, Sized)
                and len(self.cfg.shape_init_params) == 3
            )
            size = torch.as_tensor(self.cfg.shape_init_params).to(self.device)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return ((points_rand / size) ** 2).sum(
                    dim=-1, keepdim=True
                ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid

            get_gt_sdf = func
        elif self.cfg.shape_init == "sphere":
            assert isinstance(self.cfg.shape_init_params, float)
            radius = self.cfg.shape_init_params

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return (points_rand**2).sum(dim=-1, keepdim=True).sqrt() - radius

            get_gt_sdf = func
        elif self.cfg.shape_init.startswith("mesh:"):
            assert isinstance(self.cfg.shape_init_params, float)
            mesh_path = self.cfg.shape_init[5:]
            if not os.path.exists(mesh_path):
                raise ValueError(f"Mesh file {mesh_path} does not exist.")

            import trimesh

            scene = trimesh.load(mesh_path)
            if isinstance(scene, trimesh.Trimesh):
                mesh = scene
            elif isinstance(scene, trimesh.scene.Scene):
                mesh = trimesh.Trimesh()
                for obj in scene.geometry.values():
                    mesh = trimesh.util.concatenate([mesh, obj])
            else:
                raise ValueError(f"Unknown mesh type at {mesh_path}.")

            # move to center
            centroid = mesh.vertices.mean(0)
            mesh.vertices = mesh.vertices - centroid

            # align to up-z and front-x
            dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
            dir2vec = {
                "+x": np.array([1, 0, 0]),
                "+y": np.array([0, 1, 0]),
                "+z": np.array([0, 0, 1]),
                "-x": np.array([-1, 0, 0]),
                "-y": np.array([0, -1, 0]),
                "-z": np.array([0, 0, -1]),
            }
            if (
                self.cfg.shape_init_mesh_up not in dirs
                or self.cfg.shape_init_mesh_front not in dirs
            ):
                raise ValueError(
                    f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}."
                )
            if self.cfg.shape_init_mesh_up[1] == self.cfg.shape_init_mesh_front[1]:
                raise ValueError(
                    "shape_init_mesh_up and shape_init_mesh_front must be orthogonal."
                )
            z_, x_ = (
                dir2vec[self.cfg.shape_init_mesh_up],
                dir2vec[self.cfg.shape_init_mesh_front],
            )
            y_ = np.cross(z_, x_)
            std2mesh = np.stack([x_, y_, z_], axis=0).T
            mesh2std = np.linalg.inv(std2mesh)

            # scaling
            scale = np.abs(mesh.vertices).max()
            mesh.vertices = mesh.vertices / scale * self.cfg.shape_init_params
            mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

            from pysdf import SDF

            sdf = SDF(mesh.vertices, mesh.faces)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                    points_rand
                )[..., None]

            get_gt_sdf = func

            # Initialize SDF to a given shape when no weights are provided or force_shape_init is True
            optim = torch.optim.Adam(self.parameters(), lr=1e-3)
            from tqdm import tqdm

            for _ in tqdm(
                range(self.cfg.initial_steps),
                desc=f"Initializing SDF to a(n) {self.cfg.shape_init}:",
                disable=get_rank() != 0,
            ):
                points_rand = (
                    torch.rand((10000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
                )
                sdf_gt = get_gt_sdf(points_rand)
                sdf_pred = self.forward_sdf_bias(points_rand)
                loss = F.mse_loss(sdf_pred, sdf_gt)
                optim.zero_grad()
                loss.backward()
                optim.step()

        else:
            raise ValueError(
                f"Unknown shape initialization type: {self.cfg.shape_init}"
            )



        # explicit broadcast to ensure param consistency across ranks
        for param in self.parameters():
            broadcast(param, src=0)

    def get_shifted_sdf(
        self, points: Float[Tensor, "*N Di"], sdf: Float[Tensor, "*N 1"]
    ) -> Float[Tensor, "*N 1"]:
        sdf_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.sdf_bias == "ellipsoid":
            assert (
                isinstance(self.cfg.sdf_bias_params, Sized)
                and len(self.cfg.sdf_bias_params) == 3
            )
            size = torch.as_tensor(self.cfg.sdf_bias_params).to(points)
            sdf_bias = ((points / size) ** 2).sum(
                dim=-1, keepdim=True
            ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid
        elif self.cfg.sdf_bias == "sphere":
            assert isinstance(self.cfg.sdf_bias_params, float)
            radius = self.cfg.sdf_bias_params
            sdf_bias = (points**2).sum(dim=-1, keepdim=True).sqrt() - radius
        elif isinstance(self.cfg.sdf_bias, float):
            sdf_bias = self.cfg.sdf_bias
        elif self.cfg.sdf_bias.startswith("mesh"):
            sdf_bias = self.forward_sdf_bias(points=points)
        elif self.cfg.sdf_bias == "None":
            sdf_bias = 0
        else:
            raise ValueError(f"Unknown sdf bias {self.cfg.sdf_bias}")
        return sdf + sdf_bias

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        grad_enabled = torch.is_grad_enabled()

        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox.to(self.device), self.unbounded
        )  # points normalized to (0, 1)

        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        sdf = self.sdf_network(enc).view(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        output = {"sdf": sdf}
        
        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
            output.update({"features": features})

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                or self.cfg.normal_type == "finite_difference_laplacian"
            ):
                assert self.finite_difference_normal_eps is not None
                eps: float = self.finite_difference_normal_eps
                if self.cfg.normal_type == "finite_difference_laplacian":
                    offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                        [
                            [eps, 0.0, 0.0],
                            [-eps, 0.0, 0.0],
                            [0.0, eps, 0.0],
                            [0.0, -eps, 0.0],
                            [0.0, 0.0, eps],
                            [0.0, 0.0, -eps],
                        ]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 6 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    sdf_offset: Float[Tensor, "... 6 1"] = self.forward_sdf(
                        points_offset
                    )
                    sdf_grad = (
                        0.5
                        * (sdf_offset[..., 0::2, 0] - sdf_offset[..., 1::2, 0])
                        / eps
                    )
                else:
                    offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                        [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 3 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    sdf_offset: Float[Tensor, "... 3 1"] = self.forward_sdf(
                        points_offset
                    )
                    sdf_grad = (sdf_offset[..., 0::1, 0] - sdf) / eps
                normal = F.normalize(sdf_grad, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = F.normalize(normal, dim=-1)
                sdf_grad = normal
            elif self.cfg.normal_type == "analytic":
                sdf_grad = -torch.autograd.grad(
                    sdf,
                    points_unscaled,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=True,
                )[0]
                normal = F.normalize(sdf_grad, dim=-1)
                if not grad_enabled:
                    sdf_grad = sdf_grad.detach()
                    normal = normal.detach()
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update(
                {"normal": normal, "shading_normal": normal, "sdf_grad": sdf_grad}
            )
        return output

    def forward_sdf(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox.to(self.device), self.unbounded)

        sdf = self.sdf_network(
            self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        ).reshape(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        return sdf
    
    def forward_sdf_bias(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox.to(points_unscaled.device), self.unbounded)

        bias_sdf = self.sdf_bias_network(
            self.encoding_bias(points.reshape(-1, self.cfg.n_input_dims))
        ).reshape(*points.shape[:-1], 1)
        return bias_sdf
    
    def forward_color(self, points):
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
        return features

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        sdf = self.sdf_network(enc).reshape(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        deformation: Optional[Float[Tensor, "*N 3"]] = None
        if self.cfg.isosurface_deformable_grid:
            deformation = self.deformation_network(enc).reshape(*points.shape[:-1], 3)
        return sdf, deformation

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return field - threshold

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if (
            self.cfg.normal_type == "finite_difference"
            or self.cfg.normal_type == "finite_difference_laplacian"
        ):
            if isinstance(self.cfg.finite_difference_normal_eps, float):
                self.finite_difference_normal_eps = (
                    self.cfg.finite_difference_normal_eps
                )
            elif self.cfg.finite_difference_normal_eps == "progressive":
                # progressive finite difference eps from Neuralangelo
                # https://arxiv.org/abs/2306.03092
                hg_conf: Any = self.cfg.pos_encoding_config
                assert (
                    hg_conf.otype == "ProgressiveBandHashGrid"
                ), "finite_difference_normal_eps=progressive only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level
                    + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels,
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale ** (
                    current_level - 1
                )
                grid_size = 2 * self.cfg.radius / grid_res
                if grid_size != self.finite_difference_normal_eps:
                    threestudio.info(
                        f"Update finite_difference_normal_eps to {grid_size}"
                    )
                self.finite_difference_normal_eps = grid_size
            else:
                raise ValueError(
                    f"Unknown finite_difference_normal_eps={self.cfg.finite_difference_normal_eps}"
                )