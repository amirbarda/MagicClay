from dataclasses import dataclass, field

import pytorch_lightning as pl

import threestudio
from threestudio.models.exporters.base import Exporter, ExporterOutput
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import C, cleanup, load_module_weights
from threestudio.utils.saving import SaverMixin
from threestudio.utils.typing import *
from threestudio.systems.base import BaseLift3DSystem


"""
system with two base systems
"""


class HybridSystem(pl.LightningModule, Updateable, SaverMixin):
    # todo: remove this? this pattern is used elsewhere in the code, is it better\faster?
    system1: BaseLift3DSystem
    system2: BaseLift3DSystem

    def __init__(self, cfg, resumed=False):
        super().__init__()

        self.system1 = BaseLift3DSystem(
            cfg.implicit.system, resumed=resumed
        )
        self.system2 = BaseLift3DSystem(
            cfg.explicit.system, resumed=resumed
        )

        self.cfg1 = self.system1.cfg
        self.cfg2 = self.system2.cfg
        self.configure()
        self.automatic_optimization = False
        pass

    def configure(self):
        # self.system1.configure()
        # self.system2.configure()
        pass

    def configure_optimizers(self):
        optim1 = parse_optimizer(self.cfg1.optimizer, self.system1)
        optim2 = parse_optimizer(self.cfg2.optimizer, self.system2)
        ret1 = {
            "optimizer": optim1,
        }
        ret2 = {
            "optimizer": optim2,
        }
        if self.cfg1.scheduler is not None:
            ret1.update( 
                {
                    "lr_scheduler": parse_scheduler(self.cfg1.scheduler, optim1),
                }
            )
        if self.cfg2.scheduler is not None:
            ret2.update(
                {
                    "lr_scheduler": parse_scheduler(self.cfg2.scheduler, optim2),
                }
            )
            
        return ret1, ret2

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """
