from typing import Optional

import numpy as np
import pandas as pd
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

    @classmethod
    def to_representation(cls, value: int) -> float:
        if value == 1:
            value = 10
        elif value == 2:
            value = 35
        elif value == 3:
            value = 85
        elif value == 0:
            value = 0
        else:
            value = 95
        return value

    def predict(self, data):
        temp = data[0]
        data = torch.from_numpy(data[None]).float()
        data = data.to(self.device)
        probabilities = self(data)
        res_labels = torch.argmax(probabilities, dim=1)
        res = res_labels.to('cpu').detach().numpy()[0]
        if (temp < 20 and res > 2) or (temp >= 20 and res == 0):
            return self.to_representation(1)
        else:
            return self.to_representation(res)

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

if __name__ == '__main__':
    # m = ForestFireClassifier.load_from_checkpoint('/home/ikorolkevich/git/forest_fires_prediction_service/forest_fires_classifier/grid-logs_new/ForestFireClassifier_act_ReLU_lr_0.0001_bs_64/version_0/checkpoints/last.ckpt', hparams_file='/home/ikorolkevich/git/forest_fires_prediction_service/forest_fires_classifier/grid-logs_new/ForestFireClassifier_act_ReLU_lr_0.0001_bs_64/version_0/hparams.yaml')
    # m = ForestFireClassifier.load_from_checkpoint('/home/ikorolkevich/git/forest_fires_prediction_service/forest_fires_classifier/финал/Скорость обучения 0.0001 Размер партии 64/version_0/checkpoints/model-epoch=34-val_f1score=0.96815.ckpt')
    m = ForestFireClassifier.load_from_checkpoint('/home/ikorolkevich/git/forest_fires_prediction_service/forest_fires_classifier/test/Скорость обучения 0.0001 Размер партии 32/version_0/checkpoints/model-epoch=19-val_f1score=0.91174.ckpt')
    m.to('cuda')
    print(m.device)
    m.eval()
    # 30.52,8.28,3.35,1009.0,25.0,16.0,4
    # data = pd.read_csv('/home/ikorolkevich/git/forest_fires_prediction_service/forest_fires_classifier/data_processing/new/train_l.csv').values
    # for i in data:
    #     _x = i[: -1]
    #     _y = i[-1]
    #     y = m.predict(_x)
    #     if y != _y:
    #         print(f'{y} | {_y}')
    print(m.predict(np.array([32.0,7.19,1.0,751.5,0.34,34.0])))
    # make_dot(y.mean(), params=dict(m.named_parameters())).render("attached", format="png")