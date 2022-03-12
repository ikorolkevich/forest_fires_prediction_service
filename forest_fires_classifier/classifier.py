from typing import Optional

import pytorch_lightning as pl
import torch
from torch.nn import functional as func
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, \
    TRAIN_DATALOADERS, STEP_OUTPUT
from torch import nn
from torch.optim import Adam
from torchmetrics import F1Score


class BaseForestFireClassifier(pl.LightningModule):

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.9, verbose=True, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    def predict(self, data):
        data = torch.from_numpy(data).double()
        data = data.to(self.device)
        probabilities = self(data)
        res_labels = torch.argmax(probabilities, dim=1)
        return res_labels

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.float(), y
        y_predicted = self(x)
        loss = func.cross_entropy(y_predicted, y)
        f1score = self.f1score(y_predicted, y)
        self.log("train_loss", loss)
        self.log("train_f1score", f1score, prog_bar=True, on_step=False,
                 on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x, y = batch
        x, y = x.float(), y
        y_predicted = self(x)
        loss = func.cross_entropy(y_predicted, y)
        f1score = self.f1score(y_predicted, y)
        self.log("val_loss", loss)
        self.log("val_f1score", f1score, prog_bar=True, on_step=False,
                 on_epoch=True)
        return loss


class ForestFireClassifier(BaseForestFireClassifier):

    def __init__(self, lr, num_classes, func_act):
        super().__init__()
        self.linear1 = nn.Linear(6, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 5)
        self.batch_normalization1 = nn.BatchNorm1d(128)
        self.batch_normalization2 = nn.BatchNorm1d(256)
        self.batch_normalization3 = nn.BatchNorm1d(128)
        self.f1score = F1Score(num_classes=num_classes, threshold=0.7)
        self.lr = lr
        self.func_act = func_act()
        self.save_hyperparameters()

    def forward(self, data):
        x = self.linear1(data)
        x = self.func_act(x)
        x = self.batch_normalization1(x)
        x = self.linear2(x)
        x = self.func_act(x)
        x = self.batch_normalization2(x)
        x = self.linear3(x)
        x = self.func_act(x)
        x = self.batch_normalization3(x)
        x = self.linear4(x)
        x = func.softmax(x, dim=1)
        return x


class ForestFireClassifier128Batch(BaseForestFireClassifier):

    def __init__(self, lr, num_classes, func_act):
        super().__init__()
        self.linear1 = nn.Linear(6, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 5)
        self.batch_normalization1 = nn.BatchNorm1d(128)
        self.batch_normalization2 = nn.BatchNorm1d(256)
        self.batch_normalization3 = nn.BatchNorm1d(128)
        self.f1score = F1Score(num_classes=num_classes, threshold=0.7)
        self.lr = lr
        self.func_act = func_act()
        self.save_hyperparameters()

    def forward(self, data):
        x = self.linear1(data)
        x = self.batch_normalization1(x)
        x = self.func_act(x)
        x = self.linear2(x)
        x = self.batch_normalization2(x)
        x = self.func_act(x)
        x = self.linear3(x)
        x = self.batch_normalization3(x)
        x = self.func_act(x)
        x = self.linear4(x)
        x = func.softmax(x, dim=1)
        return x


class ForestFireClassifier64(BaseForestFireClassifier):

    def __init__(self, lr, num_classes, func_act):
        super().__init__()
        self.linear1 = nn.Linear(6, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 5)
        self.batch_normalization1 = nn.BatchNorm1d(64)
        self.batch_normalization2 = nn.BatchNorm1d(128)
        self.batch_normalization3 = nn.BatchNorm1d(64)
        self.f1score = F1Score(num_classes=num_classes, threshold=0.7)
        self.lr = lr
        self.func_act = func_act()
        self.save_hyperparameters()

    def forward(self, data):
        x = self.linear1(data)
        x = self.func_act(x)
        x = self.batch_normalization1(x)
        x = self.linear2(x)
        x = self.func_act(x)
        x = self.batch_normalization2(x)
        x = self.linear3(x)
        x = self.func_act(x)
        x = self.batch_normalization3(x)
        x = self.linear4(x)
        x = func.softmax(x, dim=1)
        return x


class ForestFiresClassifierCNN(BaseForestFireClassifier):

    def __init__(self, lr, num_classes, func_act):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=(3, ))
        self.pool = nn.MaxPool1d(1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=(3,))
        self.linear1 = nn.Linear(64, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 5)
        self.batch_normalization1 = nn.BatchNorm1d(256)
        self.batch_normalization2 = nn.BatchNorm1d(128)
        self.f1score = F1Score(num_classes=num_classes, threshold=0.7)
        self.lr = lr
        self.func_act = func_act()
        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv1(x)
        x = self.func_act(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.func_act(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.func_act(x)
        x = self.batch_normalization1(x)
        x = self.linear2(x)
        x = self.func_act(x)
        x = self.batch_normalization2(x)
        x = self.linear3(x)
        x = func.softmax(x, dim=1)
        return x

