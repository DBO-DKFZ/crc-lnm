import os
import torch
import math
import timm

from PIL import Image
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Resize, Normalize

from .transforms import Transform
from .helpers import collect_relevant_kwargs

from typing import Optional, Tuple
from torch import Tensor


INTERPOLATION = {"bicubic": Image.BICUBIC}


class Backbone(nn.Module):
    """
    Feature Extractor to be used in nn.Sequential for sharding.
    """

    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        freeze: bool = True,
        unfreeze_blocks: int = 0,
        unfreeze_batchnorm: bool = False,
        dropout: float = 0,
        sequential: bool = False,
        model_dir: Optional[str] = None,
    ):
        super().__init__()
        assert backbone in (
            timm.list_models() + ["cam_res"]
        )
        self.normalize = nn.Identity()
        self.resize = nn.Identity()
        self.num_blocks = 0
            
        if "cam" in backbone:
            self.backbone = timm.create_model("resnet18")
            self.backbone.fc = nn.Conv2d(512, 1, 1)
            ckpt = torch.load(os.path.join(model_dir, "nvidia-resnet18.pt"), map_location="cpu")
            self.backbone.load_state_dict(OrderedDict(zip(self.backbone.state_dict().keys(), ckpt.values())))
            self.backbone.reset_classifier(0)
            self.num_features = self.backbone.num_features
            self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        else:
            self.backbone = timm.create_model(backbone, pretrained=pretrained)
            self.backbone.reset_classifier(0)
            self.num_features = self.backbone.num_features

            if "distilled" in backbone:
                self.backbone.head_dist = nn.Identity()
            cfg = self.backbone.default_cfg
            self.norm = Normalize(cfg["mean"], cfg["std"])
            if "vit" in backbone:
                self.resize = Resize(
                    cfg["input_size"], interpolation=INTERPOLATION[cfg["interpolation"]]
                )

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        if unfreeze_batchnorm:
            for module in self.backbone.modules():
                if (
                    isinstance(module, nn.BatchNorm1d)
                    or isinstance(module, nn.BatchNorm2d)
                    or isinstance(module, nn.LayerNorm)
                ):
                    module.requires_grad_(True)

        if unfreeze_blocks > 0:
            assert self.num_blocks >= unfreeze_blocks
            assert hasattr(self, "blocks")
            for module in self.must_unfreeze:
                module.requires_grad_(True)
            for i in range(unfreeze_blocks):
                self.blocks[i].requires_grad_(True)

        self.drop = nn.Dropout(dropout)
        self.sequential = sequential

    def forward(self, x):
        if self.sequential:
            tiles, *rest = x
            tiles = self.drop(self.backbone(tiles))
            out = (tiles, *rest)
        else:
            out = self.drop(self.backbone(x))

        return out
    