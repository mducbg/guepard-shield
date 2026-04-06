import lightning as L

from ..data_loader.datamodule import TeacherDataModule


class DatasetReshuffleCallback(L.Callback):
    """Rebuilds sequence-level shuffle order after each training epoch."""

    def __init__(self, dm: TeacherDataModule):
        self._dm = dm

    def on_train_epoch_end(self, trainer, pl_module):
        if self._dm.train_dataset is not None:
            self._dm.train_dataset.reshuffle()


class MetricsHistory(L.Callback):
    """Collects per-epoch val metrics for post-training plotting."""

    def __init__(self):
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }

    def on_train_epoch_end(self, trainer, pl_module):
        self.history["train_loss"].append(
            float(trainer.callback_metrics.get("train_loss_epoch", 0))
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        self.history["val_loss"].append(
            float(trainer.callback_metrics.get("val_loss", 0))
        )
        self.history["val_accuracy"].append(
            float(trainer.callback_metrics.get("val_accuracy", 0))
        )
        self.history["val_f1"].append(float(trainer.callback_metrics.get("val_f1", 0)))
