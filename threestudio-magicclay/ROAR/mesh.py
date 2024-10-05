
import torch
import trimesh
import numpy as np
from .util.geometry_utils import calculate_vertex_normals
from .util.func import load_mesh, normalize_vertices
from pathlib import Path


class Mesh:

    def __init__(self, geom_args = None, v = None, f = None, vn = None, v_mask = None):
        if geom_args is None:
            self.vertices = v
            self.faces = f
            self.color = None
            self.vn = vn
        else:
            self.initialize_shape(geom_args)

    @property
    def v_pos(self):
        return self.vertices

    @property
    def t_pos_idx(self):
        return self.faces

    @property
    def v_nrm(self):
        if self.vn is None:
            return calculate_vertex_normals(self.vertices, self.faces)
        else:
            return self.vn
    
    def export_mesh(self,output_path):
        trimesh.Trimesh(vertices=self.vertices.detach().cpu().numpy(),faces=self.faces.cpu().numpy()).export(output_path)

    def make_icosphere(self, radius=1, subdivision_level=2, color=None):
        icosphere = trimesh.creation.icosphere(subdivision_level, radius)
        self.vertices = torch.tensor(icosphere.vertices, dtype=torch.float, device='cuda')
        self.faces = torch.tensor(icosphere.faces, dtype=torch.long, device='cuda')
        if color is None:
            self.color = torch.ones_like(self.vertices, device='cuda') * 0.5

    def load_mesh(self, mesh_path, normalize):
        self.vertices, self.faces, _, _, valid = load_mesh(Path(mesh_path))
        # todo: allow loading colors from mesh file is exists
        self.color = torch.ones_like(self.vertices, device='cuda') * 0.5
        if normalize:
            self.vertices = normalize_vertices(self.vertices)

    def initialize_shape(self, args):
        self.vn = None
        if args.init_mesh_mode == 'sphere':
            self.make_icosphere(**args.init_mesh_info)
        elif args.init_mesh_mode == 'initial_mesh':
            self.load_mesh(**args.init_mesh_info)
        else:
            raise NotImplementedError("invalid initial mesh mode")

