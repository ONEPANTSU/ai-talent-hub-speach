import time

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy

from сnn import SimpleCNN


class LitClassifier(pl.LightningModule):
    def __init__(self, n_mels: int = 80, groups: int = 1, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleCNN(n_mels=n_mels, groups=groups)
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.epoch_start_time = 0.0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        preds = (logits.sigmoid() > 0.5).long()
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.log("epoch_time", epoch_time)
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        preds = (logits.sigmoid() > 0.5).long()
        self.val_acc.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch, _):
        x, y = batch
        logits = self(x)
        preds = (logits.sigmoid() > 0.5).long()
        self.test_acc.update(preds, y)

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_acc.compute())
        self.test_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
