import os
import json
import torch
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from datetime import datetime
from argparse import ArgumentParser

from hip import Transform, TileDataModule, TileModel, pretty_plot_confusion_matrix

MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "models")


def tile_test():

    # ------------
    # args
    # ------------

    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--resultdir", type=str, default="results")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--modeldir", type=str, default=MODEL_DIR)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = TileModel.add_args(parser)
    parser = TileDataModule.add_args(parser)
    hp = parser.parse_args()
    d = vars(hp)

    if hp.cfg is not None:
        with open(hp.cfg) as stream:
            d.update(json.load(stream))

    hp.gpus = str(hp.gpus)
    pl.seed_everything(hp.seed)

    if hp.cluster:
        hp.root = os.environ["DATASET_LOCATION"]
        hp.logdir = os.environ["EXPERIMENT_LOCATION"]
        hp.progress_bar_refresh_rate = 1000
        hp.gpus = 1

    hp.resultdir = os.path.join(hp.logdir, hp.resultdir)
    hp.ckpt = os.path.join(hp.logdir, hp.ckpt)
    hp.preds_file = os.path.join(
        os.path.dirname(hp.ckpt),
        "..",
        f"preds_{hp.test_pkl.replace(os.sep, '_')[:-4]}.json",
    )

    if hp.ckpt is None:
        raise ValueError("Checkpoint is None.")
    elif not os.path.exists(hp.ckpt):
        raise ValueError(f"Checkpoint does not exist: {hp.ckpt}")

    # ------------
    # model
    # ------------

    model = TileModel()
    model = model.load_from_checkpoint(hp.ckpt)
    model = model.eval()
    model.freeze()
    model.hparams.threshold = hp.threshold
    model.hparams.modeldir = hp.modeldir
    model.hparams.preds_file = hp.preds_file

    print(f"Threshold: {model.hparams.threshold}")

    # ------------
    # transforms
    # ------------

    test_tfms = Transform(model.normalize, model.resize, hp.test_tfms)
    model.test_tfms = test_tfms.on_gpu

    # ------------
    # data
    # ------------

    dm = TileDataModule(
        root=hp.root,
        batch_size=hp.batch_size,
        num_workers=hp.num_workers,
        test_pkl=hp.test_pkl,
        test_tfms=test_tfms.on_cpu,
        id_column=hp.id_column,
        label_column=hp.label_column,
        tile_column=hp.tile_column,
    )
    dm.setup("test")

    # ------------
    # training
    # ------------

    pass

    # ------------
    # testing
    # ------------

    trainer = pl.Trainer.from_argparse_args(hp, checkpoint_callback=False, logger=False)
    metrics = trainer.test(model, datamodule=dm, verbose=True)

    # ------------
    # saving
    # ------------


if __name__ == "__main__":
    tile_test()
