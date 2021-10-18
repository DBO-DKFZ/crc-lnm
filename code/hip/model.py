import json
import timm
import torch
import psutil
import warnings
import transformers
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from pytorch_lightning.metrics import Accuracy, ConfusionMatrix, ROC
from argparse import ArgumentParser

from .layers import Backbone
from .cm_plot import pretty_plot_confusion_matrix

from typing import Tuple, Optional


class TileModel(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 2,
        # Optimizer
        lr: float = 0.1,
        ls: float = 0.0,
        wd: float = 0.0,
        threshold: float = 0.5,
        optimizer: str = "SGD",
        schedule: str = "cosine",
        schedule_step: str = "step",
        logging_step: str = "step",
        num_warmup_steps: int = 5000,
        # Backbone
        backbone: str = "clip_vit",
        pretrained: int = 1,
        freeze: int = 1,
        unfreeze_blocks: int = 0,
        unfreeze_batchnorm: int = 0,
        dropout: float = 0.0,
        # Pooling
        mil_topk: int = 10000,
        model_dir: Optional[str] = None,
        preds_file: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if num_classes > 2:
            raise NotImplementedError("Multiclass not implemented.")

        # Metrics
        self.train_acc = Accuracy(threshold=threshold, compute_on_step=True)
        self.valid_acc = Accuracy(threshold=threshold, compute_on_step=True)
        self.test_acc = Accuracy(threshold=threshold, compute_on_step=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.train_roc = ROC(pos_label=1, compute_on_step=True)
            self.valid_roc = ROC(pos_label=1, compute_on_step=True)
            self.test_roc = ROC(pos_label=1, compute_on_step=True)
        self.valid_cm = ConfusionMatrix(
            num_classes=num_classes, threshold=threshold, compute_on_step=True
        )
        self.test_cm = ConfusionMatrix(
            num_classes=num_classes, threshold=threshold, compute_on_step=True
        )

        # Model
        self.criterion = nn.BCEWithLogitsLoss()
        self.backbone = Backbone(
            backbone,
            bool(pretrained),
            bool(freeze),
            unfreeze_blocks,
            bool(unfreeze_batchnorm),
            dropout,
            sequential=False,
            model_dir=model_dir,
        )
        self.normalize = self.backbone.normalize
        self.resize = self.backbone.resize
        self.test_tfms = nn.Identity()
        self.train_tfms = nn.Identity()

        self.classifier = nn.Linear(
            self.backbone.num_features,
            num_classes if num_classes > 2 else 1,
        )

    def label_smooth(self, y):
        return y.sub(self.hparams.ls).abs()

    def forward(self, x):
        x = self.train_tfms(x) if self.training else self.test_tfms(x)
        x = self.backbone(x)
        x = self.classifier(x)
        x = x.squeeze().float()  # Recast to float32
        return x

    def mil_topk_mean(self, t):
        if self.hparams.mil_topk > 0:
            values = t.topk(min(len(t), self.hparams.mil_topk)).values
        else:
            values = t
        return values.mean()

    @staticmethod
    def groupby_reduce(s, y, ids, mil_fn=None):
        ids_g, c = ids.unique(return_counts=True)
        idx, y_idx = ids.argsort(), c.new_zeros(len(c))
        y_idx[1:] = c[:-1].cumsum(0)
        s_g, y_g = torch.split(s[idx], c.tolist()), y[idx][y_idx]
        if mil_fn is not None:
            s_g = torch.stack(list(map(mil_fn, s_g)))
        return s_g, y_g, ids_g

    def agg_tso(self, tso, acc, roc, phase, cmm=None):
        t = {"loss": torch.stack([o["loss"] for o in tso]).mean()}
        t.update({k: torch.cat([o[k] for o in tso]) for k in tso[0] if k != "loss"})
        s, y, ids = self.groupby_reduce(t["s"], t["y"], t["ids"], self.mil_topk_mean)
        if phase == "test" and self.hparams.preds_file is not None:
            preds = {"score": s.tolist(), "label": y.tolist(), "id": ids.tolist()}
            with open(self.hparams.preds_file, "w") as stream:
                json.dump(preds, stream, indent=2)
        fpr, tpr, thresholds = roc(s, y)
        if phase == "valid":
            gmeans = (tpr * (1 - fpr)).sqrt()
            self.hparams.threshold = thresholds[gmeans.argmax()].item()
        acc.threshold = self.hparams.threshold
        slide_loss = F.binary_cross_entropy(s.float(), y.float())
        step = (
            self.current_epoch
            if self.hparams.logging_step == "epoch"
            else self.global_step
        )
        if cmm is not None:
            cmm.threshold = self.hparams.threshold
            cmm_results = cmm(s.float(), y).cpu().numpy()
            if phase != "test":
                self.logger.experiment.add_figure(
                    f"{phase}/confusion_matrix",
                    pretty_plot_confusion_matrix(cmm_results),
                    global_step=step,
                )
        metrics = {
            f"{phase}/loss_epoch_tile": t["loss"].item(),
            f"{phase}/acc": acc(s.float(), y).item(),
            f"{phase}/auroc": torch.trapz(tpr, fpr).item(),
            f"{phase}/best_thresh": self.hparams.threshold,
            f"{phase}/loss_epoch_slide": slide_loss.item(),
        }

        if phase == "test":
            tn, fp, fn, tp = cmm_results.ravel()
            tpr = tp / (tp + fn)
            tnr = tn / (tn + fp)
            bacc = (tpr + tnr) / 2
            metrics["test/sensitivity"] = tpr
            metrics["test/specificity"] = tnr
            metrics["test/balanced_accuracy"] = bacc

        return metrics

    def training_step(self, batch, batch_idx):
        x, (y, ids) = batch
        logits = self(x)
        loss = self.criterion(logits, self.label_smooth(y))
        self.log(
            "sys/ram_percent",
            psutil.virtual_memory().percent,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        self.log(
            "train/loss_step_tile", loss, on_step=True, on_epoch=False, logger=True
        )
        return {
            "loss": loss,
            "s": logits.detach().sigmoid(),
            "y": y.detach(),
            "ids": ids.detach(),
        }

    def training_step_end(self, tso):
        tso["loss"] = tso["loss"].mean()
        return tso

    def training_epoch_end(self, tso):
        m = self.agg_tso(tso, self.train_acc, self.train_roc, "train")
        step = (
            self.current_epoch
            if self.hparams.logging_step == "epoch"
            else self.global_step
        )
        self.logger.agg_and_log_metrics(m, step=step)

    def validation_step(self, batch, batch_idx):
        x, (y, ids) = batch
        logits = self(x)
        loss = self.criterion(logits, self.label_smooth(y)).detach()
        return {
            "loss": loss,
            "s": logits.detach().sigmoid(),
            "y": y.detach(),
            "ids": ids.detach(),
        }

    def validation_step_end(self, tso):
        return self.training_step_end(tso)

    def validation_epoch_end(self, tso):
        m = self.agg_tso(
            tso, self.valid_acc, self.valid_roc, "valid", cmm=self.valid_cm
        )
        step = (
            self.current_epoch
            if self.hparams.logging_step == "epoch"
            else self.global_step
        )
        self.logger.agg_and_log_metrics(m, step=step)
        self.log_dict({k.replace("/", "_"): v for k, v in m.items()}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, tso):
        return self.validation_step_end(tso)

    def test_epoch_end(self, tso):
        m = self.agg_tso(tso, self.test_acc, self.test_roc, "test", cmm=self.test_cm)
        self.log_dict(m)

    def configure_optimizers(self):
        kwargs = {
            "params": self.parameters(),
            "lr": self.hparams.lr,
            "weight_decay": self.hparams.wd,
        }
        if self.hparams.optimizer == "SGD":
            opt = torch.optim.SGD(**kwargs, momentum=0.9)
        elif self.hparams.optimizer == "AdamW":
            opt = torch.optim.AdamW(**kwargs)
        elif self.hparams.optimizer == "Adam":
            opt = torch.optim.Adam(**kwargs)
        else:
            raise ValueError(f"Unsupported optimizer {self.hparams.optimizer}")

        if self.hparams.schedule_step == "step":
            num_warmup_steps = self.hparams.num_warmup_steps
            num_training_steps = self.trainer.max_steps

        elif self.hparams.schedule_step == "epoch":
            num_epoch_steps = len(self.train_dataloader())
            num_warmup_steps = num_epoch_steps * self.hparams.num_warmup_steps
            num_training_steps = num_epoch_steps * self.trainer.max_epochs

        kwargs = {
            "optimizer": opt,
            "num_warmup_steps": num_warmup_steps,
            "num_training_steps": num_training_steps,
        }
        shd = {"scheduler": None, "interval": "step"}
        if self.hparams.schedule == "linear":
            shd["scheduler"] = transformers.get_linear_schedule_with_warmup(**kwargs)
        elif self.hparams.schedule == "cosine":
            shd["scheduler"] = transformers.get_cosine_schedule_with_warmup(**kwargs)
        elif self.hparams.schedule == "constant":
            kwargs.pop("num_training_steps")
            shd["scheduler"] = transformers.get_constant_schedule_with_warmup(**kwargs)
        else:
            raise ValueError(f"Unsupported schedule {self.hparams.schedule}")
        return [opt], [shd]

    @staticmethod
    def add_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_classes", type=int, default=2)
        # Optimizer
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--ls", type=float, default=0.1)
        parser.add_argument("--wd", type=float, default=0.0)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--optimizer", type=str, default="AdamW")
        parser.add_argument("--schedule", type=str, default="cosine")
        parser.add_argument("--schedule_step", type=str, default="step")
        parser.add_argument("--logging_step", type=str, default="step")
        parser.add_argument("--num_warmup_steps", type=int, default=5000)
        # Backbone
        parser.add_argument("--backbone", type=str, default="clip_vit")
        parser.add_argument("--pretrained", type=int, default=1)
        parser.add_argument("--freeze", type=int, default=1)
        parser.add_argument("--unfreeze_blocks", type=int, default=0)
        parser.add_argument("--unfreeze_batchnorm", type=int, default=0)
        parser.add_argument("--dropout", type=float, default=0.3)
        # Pooling
        parser.add_argument("--mil_topk", type=int, default=10000)
        return parser
