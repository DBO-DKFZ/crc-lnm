import os
import json
import torch
import pytorch_lightning as pl

from datetime import datetime
from argparse import ArgumentParser

from hip import Transform, TileDataModule, TileModel

MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "models")


def tile_train():

    # ------------
    # args
    # ------------

    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
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

    ### If you want to save your cfg ###
    # with open("my_cfg.json", 'w') as stream:
    #    out = {k: v for k, v in d.items() if not callable(v)}
    #    json.dump(out, stream, sort_keys=True, indent=2)
    #    print('INFO: CFG WAS SAVED!')
    #    return

    hp.name = hp.name or os.path.basename(hp.cfg).split(".")[0]
    hp.gpus = str(hp.gpus)
    pl.seed_everything(hp.seed)

    if hp.cluster:
        hp.root = os.environ["DATASET_LOCATION"]
        hp.logdir = os.environ["EXPERIMENT_LOCATION"]
        hp.progress_bar_refresh_rate = 1000
        hp.gpus = 1

    # ------------
    # model
    # ------------

    model = TileModel(
        num_classes=hp.num_classes,
        # Optimizer
        lr=hp.lr,
        ls=hp.ls,
        wd=hp.wd,
        threshold=hp.threshold,
        optimizer=hp.optimizer,
        schedule=hp.schedule,
        schedule_step=hp.schedule_step,
        logging_step=hp.logging_step,
        num_warmup_steps=hp.num_warmup_steps,
        # Backbone
        backbone=hp.backbone,
        pretrained=hp.pretrained,
        freeze=hp.freeze,
        unfreeze_blocks=hp.unfreeze_blocks,
        unfreeze_batchnorm=hp.unfreeze_batchnorm,
        dropout=hp.dropout,
        # Pooling
        mil_topk=hp.mil_topk,
        model_dir=hp.modeldir,
    )

    # ------------
    # transforms
    # ------------

    train_tfms = Transform(model.normalize, model.resize, hp.train_tfms)
    test_tfms = Transform(model.normalize, model.resize, hp.test_tfms)
    model.train_tfms = train_tfms.on_gpu
    model.test_tfms = test_tfms.on_gpu

    # ------------
    # data
    # ------------

    dm = TileDataModule(
        root=hp.root,
        batch_size=hp.batch_size,
        num_workers=hp.num_workers,
        train_pkl=hp.train_pkl,
        valid_pkl=hp.valid_pkl,
        test_pkl=None,
        train_tfms=train_tfms.on_cpu,
        test_tfms=test_tfms.on_cpu,
        id_column=hp.id_column,
        label_column=hp.label_column,
        tile_column=hp.tile_column,
        split_by=hp.split_by,
        valid_perc=hp.valid_perc,
        distributed="ddp" in (hp.accelerator or ""),
    )
    dm.setup("fit")

    # ------------
    # training
    # ------------

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = os.path.join(hp.logdir, hp.name)
    ckpt_dir = os.path.join(log_path, "ckpts")
    ckpt_file = f"{now}-{hp.backbone}"
    ckpt_file += "-{step:05d}-{valid_auroc:.2f}"
    ckpt_file += "-{valid_loss_epoch_tile:.2f}-{valid_loss_epoch_slide:.2f}"

    lr_monitor = pl.callbacks.lr_monitor.LearningRateMonitor()
    logger = pl.loggers.TensorBoardLogger(log_path, name=None)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid_loss_epoch_slide",
        dirpath=ckpt_dir,
        filename=ckpt_file,
        save_top_k=1,
        mode="min",
    )
    print("\nDeterminsitic:", hp.deterministic, "\n")
    trainer = pl.Trainer.from_argparse_args(
        hp,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        resume_from_checkpoint=hp.resume_from_checkpoint,
    )

    trainer.fit(model, datamodule=dm)

    # ------------
    # testing
    # ------------

    pass


if __name__ == "__main__":
    tile_train()
