import random

from torch import nn
from collections import OrderedDict
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from typing import List, Optional


class Rot(nn.Module):
    """Rotate PIL Image by one of the given angles."""

    def __init__(self, angles: List[int] = [0, 90, 180, 270]):
        super().__init__()
        self.angles = tuple(angles)

    def forward(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

    def __repr__(self):
        return f"{self.__class__.__name__}(angles={self.angles})"


class Scale(nn.Module):
    """Replaces scaling to [0, 1] normally done by ToTensor."""

    def __init__(self, factor: int = 255):
        super().__init__()
        self.factor = factor

    def forward(self, tensor):
        return tensor / self.factor

    def __repr__(self):
        return f"{self.__class__.__name__}(factor={self.factor})"


class Transform(nn.Module):
    """Assemble transforms from str to ops."""

    def __init__(
        self,
        normalize: Optional[nn.Module] = None,
        resize: Optional[nn.Module] = None,
        tfms: Optional[str] = "",
    ):
        super().__init__()
        if normalize is not None:
            normalize.inplace = True
        ops = {
            "normalize": normalize,
            "resize": resize or nn.Identity(),
            "rotate": Rot([0, 90, 180, 270]),
            "scale": Scale(),
            "vflip": T.RandomVerticalFlip(p=0.5),
            "hflip": T.RandomHorizontalFlip(p=0.5),
            "autoaugment": T.AutoAugment(),
            "simplejitter": T.ColorJitter(
                brightness=0.25, contrast=0.75, saturation=0.25, hue=0
            ),
            "colorjitter": T.ColorJitter(
                brightness=0.25, contrast=0.75, saturation=0.25, hue=0.04
            ),
            "randomresizecrop": T.RandomResizedCrop(
                224, scale=(0.5, 1.0), ratio=(0.75, 1.33), interpolation=3
            ),
            "grayscale": T.Grayscale(num_output_channels=3),
            "randomgrayscale": T.RandomGrayscale(p=0.5),
        }
        on_cpu = []
        on_gpu = []

        if len(tfms) > 0:
            tfms = tfms.replace(" ", "")
            on_cpu, on_gpu = [t.strip(",").split(",") for t in tfms.split("gpu")]
            on_cpu = [] if (len(on_cpu) == 1) and (on_cpu[0] == "") else on_cpu
            on_gpu = [] if (len(on_gpu) == 1) and (on_gpu[0] == "") else on_gpu

        self.on_cpu = nn.Sequential(OrderedDict([[t, ops[t]] for t in on_cpu]))
        self.on_gpu = nn.Sequential(OrderedDict([[t, ops[t]] for t in on_gpu]))

    def forward(self, x):
        x = self.on_cpu(x)
        x = self.on_gpu(x)
        return x
