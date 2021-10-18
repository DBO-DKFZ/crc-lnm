import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch import nn
from fastcore.foundation import L
from fastcore.basics import store_attr
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.io import read_image
from argparse import ArgumentParser

from .helpers import balanced_train_test_split

# Typing
from typing import Callable, List, Optional, Sequence, Tuple, Union
from numpy import ndarray as Array
from torch import Generator, Tensor
from pandas import DataFrame

ArrayLike = Union[Tensor, Array, List, Tuple, Sequence]


class TileSampler(Sampler):
    """Samples tiles while balancing labels. If num_max_tiles is specified,
    a new subset will be drawn after every epoch.

    Args:
        tile_labels (sequence): Label of each tile.
        tile_slide_ids (sequence): SlideID of each tile.
        num_max_tiles (int): Maximum number of tiles per slide.
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    """

    def __init__(
        self,
        tile_labels: ArrayLike,
        tile_slide_ids: ArrayLike,
        num_max_tiles: Optional[int] = None,
        replacement: bool = True,
        generator: Optional[Generator] = None,
    ):

        self.labels = torch.as_tensor(tile_labels)
        self.slide_ids = torch.as_tensor(tile_slide_ids)
        self.tiles = [
            torch.where(self.slide_ids == i)[0] for i in self.slide_ids.unique()
        ]
        self.weights = torch.zeros_like(self.labels).double()
        self.max_tiles = num_max_tiles
        self.replacement = replacement
        self.generator = generator
        self.class_weights = None
        self.num_samples = None
        self.update_weights()

    def update_weights(self):
        if self.max_tiles is not None and self.max_tiles != 0:
            subset_idx = torch.cat(
                [
                    tiles[torch.randperm(len(tiles))[: self.max_tiles]]
                    for tiles in self.tiles
                ]
            )
            subset_labels = self.labels[subset_idx]
        else:
            subset_idx = torch.arange(len(self.labels))
            subset_labels = self.labels

        if self.class_weights is None:
            label_counts = subset_labels.unique(return_counts=True)[1]
            self.class_weights = 1 / label_counts.double()
            self.num_samples = label_counts.sum().int().item()

        self.weights.zero_()
        self.weights[subset_idx] = self.class_weights[subset_labels]

    def __iter__(self):
        self.update_weights()
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=self.generator
        )
        return iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples


class TileDataSet(Dataset):
    def __init__(
        self,
        df: DataFrame,
        id_column: str = "slide_name",
        tile_column: str = "tiles",
        label_column: str = "label",
        tile_tfms: Callable = nn.Identity(),
        label_tfms: Callable = nn.Identity(),
    ):
        attrs = list(locals())
        attrs.remove("self")
        store_attr(names=",".join(attrs))

        assert len(self.df) == len(self.df[id_column].unique())
        self.id_map = {i: s for i, s in enumerate(self.df[id_column])}
        self.id_map.update(dict(map(reversed, self.id_map.items())))  # Reversible

        self.samples = L()
        self.labels = L()
        empty = []
        for i, (paths, label) in enumerate(
            zip(self.df[tile_column], self.df[label_column].astype(int))
        ):
            assert len(paths) >= 0
            labels = [[label, i]] * len(paths)
            self.samples += zip(paths, labels)
            self.labels += labels
        self.stats = {
            "tiles": np.unique(self.labels.itemgot(0), return_counts=True),
            "slides": np.unique(self.df[label_column], return_counts=True),
            "n_tiles": self.df[tile_column]
            .map(len)
            .agg({"mean": np.mean, "median": np.median, "min": np.min, "max": np.max}),
        }
        self.num_classes = len(self.labels.itemgot(0).unique())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.tile_tfms(read_image(path)), self.label_tfms(label)


class TileDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int = 512,
        num_workers: int = 4,
        train_pkl: Optional[str] = None,
        valid_pkl: Optional[str] = None,
        test_pkl: Optional[str] = None,
        train_tfms: Callable = nn.Identity(),
        test_tfms: Callable = nn.Identity(),
        valid_perc: float = 0.2,
        id_column: str = "slide_name",
        tile_column: str = "tiles",
        label_column: str = "label",
        split_by: str = "label",
        distributed: bool = False,
    ):
        super().__init__()
        attrs = list(locals())
        attrs.remove("self")
        store_attr(names=",".join(attrs))

        self.root = os.path.abspath(self.root)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Make paths absolute
        def rootify(paths):
            def inner(path):
                return os.path.join(self.root, path)

            return list(map(inner, paths))

        if stage == "fit" or stage is None:
            train_pkl = os.path.join(self.root, self.train_pkl)
            if self.valid_pkl is None:
                self.train_df, self.valid_df = balanced_train_test_split(
                    pd.read_pickle(train_pkl), self.split_by, self.valid_perc
                )
            else:
                valid_pkl = os.path.join(self.root, self.valid_pkl)
                self.train_df = pd.read_pickle(train_pkl)
                self.valid_df = pd.read_pickle(valid_pkl)

            self.train_df[self.tile_column] = self.train_df[self.tile_column].map(
                rootify
            )
            self.valid_df[self.tile_column] = self.valid_df[self.tile_column].map(
                rootify
            )
            self.train_ds = TileDataSet(
                df=self.train_df,
                id_column=self.id_column,
                tile_column=self.tile_column,
                label_column=self.label_column,
                tile_tfms=self.train_tfms,
            )
            self.valid_ds = TileDataSet(
                df=self.valid_df,
                id_column=self.id_column,
                tile_column=self.tile_column,
                label_column=self.label_column,
                tile_tfms=self.test_tfms,
            )

            self.dims = tuple(self.train_ds[0][0].shape)
            self.num_classes = self.train_ds.num_classes

        if stage == "test" or stage is None:
            self.test_df = pd.read_pickle(os.path.join(self.root, self.test_pkl))
            self.test_df[self.tile_column] = self.test_df[self.tile_column].map(rootify)
            self.test_ds = TileDataSet(
                df=self.test_df,
                id_column=self.id_column,
                tile_column=self.tile_column,
                label_column=self.label_column,
                tile_tfms=self.test_tfms,
            )
            self.dims = tuple(self.test_ds[0][0].shape)
            self.num_classes = self.test_ds.num_classes

    def train_dataloader(
        self,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        distributed: bool = False,
    ):
        sampler = TileSampler(
            self.train_ds.labels.itemgot(0),
            self.train_ds.labels.itemgot(1),
            num_max_tiles=int(self.train_ds.stats["n_tiles"]["median"]),
        )

        if distributed or self.distributed:
            sampler = DistributedSamplerWrapper(sampler)

        return DataLoader(
            self.train_ds,
            shuffle=False,
            sampler=sampler,
            batch_size=batch_size or self.batch_size,
            drop_last=True,
            num_workers=num_workers or self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(
        self, num_workers: Optional[int] = None, batch_size: Optional[int] = None
    ):
        return DataLoader(
            self.valid_ds,
            shuffle=False,
            batch_size=batch_size or self.batch_size,
            num_workers=num_workers or self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(
        self, num_workers: Optional[int] = None, batch_size: Optional[int] = None
    ):
        return DataLoader(
            self.test_ds,
            shuffle=False,
            batch_size=batch_size or self.batch_size,
            num_workers=num_workers or self.num_workers,
            pin_memory=True,
        )

    @staticmethod
    def add_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--root", type=str, default=None)
        parser.add_argument("--batch_size", type=int, default=384)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--train_pkl", type=str, default=None)
        parser.add_argument("--valid_pkl", type=str, default=None)
        parser.add_argument("--test_pkl", type=str, default=None)
        parser.add_argument(
            "--train_tfms",
            type=str,
            default="gpu,resize,rotate,vflip,hflip,scale,normalize",
        )
        parser.add_argument(
            "--test_tfms",
            type=str,
            default="gpu,resize,scale,normalize",
        )
        parser.add_argument("--valid_perc", type=float, default=0.2)
        parser.add_argument("--id_column", type=str, default="slide_name")
        parser.add_argument("--tile_column", type=str, default="tiles")
        parser.add_argument("--label_column", type=str, default="label")
        parser.add_argument("--split_by", type=str, default="label")
        return parser
