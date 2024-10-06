import math
import random
from dataclasses import dataclass
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.dataset import T_co
import threestudio
from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
import copy
from threestudio.utils.typing import *

class HybridRandomCameraIterableDataset(IterableDataset):
    
    def __init__(self, dm1,dm2):

        self.dataset1 = dm1.train_dataset
        self.dataset2 = dm2.train_dataset 

    def __iter__(self):
        while True:
            yield {}

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.dataset1.update_step(epoch, global_step, on_load_weights)

    def collate(self, batch) -> Dict[str, Any]:
        res1 = self.dataset1.collate(batch)
        focal_length: Float[Tensor, "B"] = 0.5 * self.dataset2.height / torch.tan(0.5 * res1["fovy"])
        directions: Float[Tensor, "B H W 3"] = self.dataset2.directions_unit_focal[
            None, :, :, :
        ].repeat(self.dataset1.batch_size, 1, 1, 1)
        
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(
            directions, res1["c2w"], keepdim=True, normalize=self.dataset1.cfg.rays_d_normalize
        )
        res2 = copy.deepcopy(res1)
        res2["height"] = self.dataset2.height
        res2["width"] = self.dataset2.width
        res2["rays_o"] = rays_o
        res2["rays_d"] = rays_d
        
        return [res1, res2]


class HybridRandomCameraDataset(Dataset):

    def __init__(self, dm1, dm2, split: str):
        if split == 'val':
            self.dataset1 = dm1.val_dataset
        elif split == 'test':
            self.dataset1 = dm1.test_dataset      

    def __len__(self):
        return self.dataset1.n_views

    def __getitem__(self, index):
        item1 = self.dataset1.__getitem__(index)
        return [item1, item1]

    def collate(self, batch) -> Dict[str, Any]:
        res1 = self.dataset1.collate([batch[0][0]])
        return [res1, res1]


@register("hybrid-random-camera-datamodule")
class HybridRandomCameraDataModule(pl.LightningDataModule):    
    
    # cfg: RandomMultiviewCameraDataModuleConfig
    
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.dm1 = threestudio.find(cfg.implicit.data_type)(cfg.implicit.data)
        self.dm2 = threestudio.find(cfg.explicit.data_type)(cfg.explicit.data)

    def setup(self, stage=None) -> None:
        self.dm1.setup(stage)
        self.dm2.setup(stage)
        if stage in [None, "fit"]:
            self.train_dataset = HybridRandomCameraIterableDataset(self.dm1, self.dm2)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = HybridRandomCameraDataset(self.dm1, self.dm2, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = HybridRandomCameraDataset(self.dm1, self.dm2, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
