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
        hp.resultdir = os.path.join(hp.logdir, hp.resultdir)
        hp.ckpt = os.path.join(hp.logdir, hp.ckpt)
        hp.progress_bar_refresh_rate = 1000
        hp.gpus = 1

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
    cm = model.test_cm.compute().cpu().numpy()
    fig = pretty_plot_confusion_matrix(cm)

    # ------------
    # saving
    # ------------
    save_file = os.path.join(hp.logdir, hp.results)

    metrics = metrics[0]
    test_data = os.path.basename(hp.test_pkl)
    train_data = os.path.basename(hp.train_pkl)
    ckpt_name = os.path.basename(hp.ckpt)
    metrics["checkpoint"] = ckpt_name
    metrics["test_data"] = test_data
    metrics["train_data"] = train_data
    metrics["hypothesis"] = train_data[:2]
    metrics["matched_train"] = "matched" in train_data
    metrics["#tile_neg"], metrics["#tile_pos"] = dm.test_ds.stats["tiles"][1]
    metrics["#slide_neg"], metrics["#slide_pos"] = dm.test_ds.stats["slides"][1]
    plt.savefig(os.path.join(hp.logdir, f"{test_data}_{ckpt_name}.png"), dpi=300)
    if os.path.exists(save_file):
        df = pd.read_csv(save_file)
    else:
        df = pd.DataFrame()
    df = df.append(metrics, ignore_index=True)
    df.to_csv(save_file, index=False, na_rep="NaN")


if __name__ == "__main__":
    tile_test()
