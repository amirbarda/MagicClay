from dataclasses import dataclass, field

import torch
import torch_scatter
import configargparse
import threestudio
from ..ROAR.util.topology_utils import calculate_vertex_incident_scalar, calc_edges
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from ..ROAR.mesh_optimizer.mesh_opt import MeshOptimizer
from threestudio.models.geometry.base import (
    BaseExplicitGeometry,
    BaseGeometry,
    contract_to_unisphere,
)
from ..ROAR.mesh import Mesh
from ..ROAR.configs.config import Config
import numpy as np
from threestudio.models.networks import get_encoding, get_mlp
import torch.nn.functional as F
from pathlib import Path
import copy

def read_mtlfile(fname):
    materials = {}
    with open(fname) as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        if line:
            prefix, data = line.split(' ', 1)
            if 'newmtl' in prefix:
                material = {}
                materials[data] = material
            elif materials:
                if len(data.split(' ')) > 1:
                    material[prefix] = tuple(float(d) for d in data.split(' '))
                else:
                    if data.isdigit():
                        try:
                            material[prefix] = int(data)
                        except ValueError:
                            material[prefix] = float(data)
                    else:
                        material[prefix] = data

    return materials
    

@threestudio.register("dynamic-mesh")
class DynamicMesh(BaseExplicitGeometry):

    @dataclass
    class Config(BaseExplicitGeometry.Config):
        optimize_normal_map: bool = False
        encode_initial_mtl: bool = False
        initial_mtl_path: str = ""
        initial_mesh_path: str = ""
        allowed_vertices_path: str = ""
        animation_weights_path: str = ""
        vertex_colors_path: str = ""
        appearance_embedding_steps: int = 500
        initial_fit_steps: int = 0
        config_path : str = ""
        config_extras : str = ""
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

    def get_mesh(self):
        self.mesh.allowed_vertices = self.allowed_vertices
        return self.mesh    

    
    def set_lr(self, lr):
        self.mesh_optimizer._lr = lr


    def get_supersampled_mesh(self, orient_normals_strategy = 'smooth', offset = 0):
        v = self.mesh.vertices
        f = self.mesh.faces
        v = v.unsqueeze(0)
        f = f.unsqueeze(0)
        self.mesh_optimizer.reg_sampler.sample_regular(v,f, orient_normals_strategy=orient_normals_strategy, offset = offset)
        v_supersampled = self.mesh_optimizer.reg_sampler.sampled_vertices
        f_supersampled = self.mesh_optimizer.reg_sampler.sampled_faces
        vn_supersampled = self.mesh_optimizer.reg_sampler.sampled_vertices_normals
        mesh = Mesh(v = v_supersampled, f = f_supersampled, vn = vn_supersampled)
        allowed_faces, _ = self.mesh_optimizer.get_allowed_faces_and_edges()
        mesh.allowed_vertices = allowed_faces[self.mesh_optimizer.reg_sampler.face_ids]
        return mesh

    def forward_color_default(self,points):
        return torch.ones_like(points, device=points.device)*0.5

    def forward_color_network(self, points):
        
        points = contract_to_unisphere(
            points, self.bbox.to(points.device), False
        )  # points normalized to (0, 1)
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
        return features

    # todo: handle loading if started with resume flag
    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        cfg_path = self.cfg['config_path']
        config_extras = self.cfg['config_extras']
        # parse mesh config
        options = Config().parse(args='--config {} {}'.format(cfg_path, config_extras))
        options.device = self.device
        self.has_animation = False
        animation_weight_matrix = None
        if self.cfg.initial_mesh_path != "":
            options.init_mesh_mode = 'initial_mesh'
            options.init_mesh_info = {'mesh_path' : self.cfg.initial_mesh_path, 'normalize' : False}
        self.mesh = Mesh(options)

        if self.cfg.allowed_vertices_path == "None":
            allowed_vertices = torch.zeros(self.mesh.vertices.shape[0], dtype = torch.bool, device=options.device)
        elif self.cfg.allowed_vertices_path != "":
            allowed_vertices_ids  = torch.tensor(np.loadtxt(self.cfg.allowed_vertices_path),device=self.device, dtype=torch.long)
            allowed_vertices = torch.zeros(self.mesh.vertices.shape[0], dtype = torch.bool, device=options.device)
            allowed_vertices[allowed_vertices_ids] = True
        else:
            allowed_vertices = torch.ones(self.mesh.vertices.shape[0], dtype = torch.bool, device=options.device)
        if self.cfg.animation_weights_path != "":
            self.has_animation = True
            animation_weight_matrix = torch.tensor(np.loadtxt(self.cfg.animation_weights_path),device=self.device, dtype = torch.float)

        self.original_mesh = copy.deepcopy(self.mesh)  
        self.original_mesh.allowed_vertices = allowed_vertices
        options.reconstruction_mode = 'sdf' # TODO: move to config file
        options.max_faces = self.mesh.faces.shape[0] + options.max_faces # add max faces to current amount of faces (negative amount means simplification)
        self.mesh_optimizer = MeshOptimizer(self.mesh.vertices, self.mesh.faces, options, allowed_vertices = allowed_vertices, extra_vertex_attributes = animation_weight_matrix)
        self.mesh.vertices = self.mesh_optimizer.vertices
        self.mesh.faces = self.mesh_optimizer.faces
        self.allowed_vertices = self.mesh_optimizer.allowed_vertices

        # if initial texture exists, train network to mimic the colors
        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        ).to(self.device)
        self.feature_network = get_mlp(
            self.encoding.n_output_dims,
            self.cfg.n_feature_dims,
            self.cfg.mlp_network_config,
        ).to(self.device)
        
        if self.cfg.encode_initial_mtl:
            import imageio as iio
            from PIL import Image

            from ..ROAR.sampling.regular_sampler import RegularSampler
            import igl
            points_colors= None      
            if self.cfg.initial_mtl_path != '' and Path(self.cfg.initial_mtl_path).exists() :
                material = read_mtlfile(self.cfg.initial_mtl_path)
                parent_path = Path(self.cfg.initial_mtl_path).parent
                texture_image = iio.imread(Path(parent_path,material[next(iter(material))]['map_Kd']))
                # supersample points on the mesh
                _,tc,_,_,ftc,_ = igl.read_obj(self.cfg.initial_mesh_path)
                v = self.mesh.vertices.detach()
                tc = torch.tensor(tc, device=self.device, dtype = torch.float)
                f = self.mesh.faces
                ftc = torch.tensor(ftc, device=self.device, dtype = torch.long)
                reg_sampler = RegularSampler(device = self.device, delta=0.002, max_sampling_n=20)
                reg_sampler.sample_regular(v.unsqueeze(0), f.unsqueeze(0))
                self.bbox = torch.stack([v.min(dim=0)[0], v.max(dim=0)[0]], dim=0)
                points = reg_sampler.sampled_vertices #v[f].mean(axis=1)
                # get barycentric coordinates of points
                bc = reg_sampler.barycentric_coords #torch.ones_like(f, device = self.device)/3
                ftc = ftc[reg_sampler.face_ids]
                # get texture coordinate of each point
                tc = tc.clip(0,1)
                points_tc = torch.sum(tc[ftc]*bc[...,None], dim=1).clip(0)
                points_tc[:,0] *= texture_image.shape[0]-1
                points_tc[:,1] *= texture_image.shape[1]-1
                points_tc = points_tc.round().to(torch.long).cpu().numpy()
                # sample color on image
                image = Image.fromarray(texture_image)
                image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
                texture_image = np.array(image)
                points_colors = texture_image[points_tc[:,1], points_tc[:,0]]
                sampled_points = reg_sampler.sampled_vertices
                # train appearance network to mimic this

                points_colors = torch.tensor(points_colors, device = self.device, dtype = torch.float)/255
                points_colors = points_colors.clip(0,1)

            elif self.cfg.vertex_colors_path != '' and Path(self.cfg.vertex_colors_path).exists() :
                    points_colors = torch.tensor(np.load(self.cfg.vertex_colors_path), device=self.device, dtype = torch.float).clip(0,1)
                    _,tc,_,_,ftc,_ = igl.read_obj(self.cfg.initial_mesh_path)
                    v = self.mesh.vertices.detach()
                    f = self.mesh.faces
                    sampled_points = v

            if points_colors is not None:
                self.forward_color = self.forward_color_network
                num_training_segments = 3
                num_points = sampled_points.shape[0]
                optim = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
                from tqdm import tqdm
                for i in (pbar:=tqdm(
                    range(self.cfg.appearance_embedding_steps),
                    desc=f"Initializing intial appearance:",
                    #disable=get_rank() != 0,
                )):
                    segment = i % num_training_segments
                    points = sampled_points[(num_points*segment//num_training_segments): (num_points*(segment+1)//num_training_segments)]
                    color_gt = points_colors[(num_points*segment//num_training_segments): (num_points*(segment+1)//num_training_segments)]
                    color_pred = self.forward_color(points)
                    loss = F.mse_loss(color_pred, color_gt[:,:3])
                    pbar.set_postfix_str('loss: {:.4E}'.format(loss.item()))
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
            else:           
                self.forward_color = self.forward_color_default

        else:
            self.forward_color = None           

    def export_mesh(self,path):
        self.mesh.export_mesh(path)

    def get_animation_weight_matrix(self):
        return self.mesh_optimizer._vertices_etc[:,9:]

    def calculate_laplacian_of_vertices(self):
            vertices = self.mesh_optimizer.vertices
            faces = self.mesh_optimizer.faces
            edges, _ = calc_edges(faces, with_dummies=False) 
            edge_smooth = vertices[edges] # E,2,S
            neighbor_smooth = torch.zeros_like(vertices)  # V,S - mean of 1-ring vertices
            torch_scatter.scatter_mean(src=edge_smooth.flip(dims=[1]).reshape(edges.shape[0] * 2, -1), index=edges.reshape(edges.shape[0] * 2, 1),
                                   dim=0, out=neighbor_smooth)
            laplace = vertices - neighbor_smooth[:, :3]  # laplace vector
            return laplace

    def faces_for_freeze_loss(self, rounds = 0):
        allowed_vertices = self.mesh_optimizer.allowed_vertices.clone()
        faces_to_freeze = allowed_vertices[self.mesh_optimizer._faces].sum(dim=-1) == 0
        
        for _ in range(rounds):
            vertex_freeze_score = calculate_vertex_incident_scalar(self.mesh_optimizer._vertices.shape[0], self.mesh_optimizer._faces, allowed_vertices[self.mesh_optimizer._faces].sum(dim=-1).float(), avg=False)
            allowed_vertices = vertex_freeze_score.squeeze(-1) > 0
            faces_to_freeze = allowed_vertices[self.mesh_optimizer._faces].sum(dim=-1) == 0
        return faces_to_freeze
    

    def step(self):
        vertices, _ = self.mesh_optimizer.step()
        self.mesh.vertices = vertices
        self.mesh.vn = None
    
    def remesh(self, time_step):
        vertices, faces, debug_info = self.mesh_optimizer.remesh(time_step)
        self.mesh.vertices = vertices
        self.mesh.faces = faces
        self.allowed_vertices = self.mesh_optimizer.allowed_vertices
        self.mesh.vn = None
        
    def zero_grad(self):
        self.mesh.vertices.grad = None
        self.mesh.vn = None
        self.mesh_optimizer.zero_grad()
