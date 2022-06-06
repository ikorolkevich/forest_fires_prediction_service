import random

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, \
    TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co


class ForestFiresLinearDataset(Dataset):

    def __init__(self, path_to_csv: str, label_col: str):
        self.data = pd.read_csv(path_to_csv)
        columns = list(self.data.columns)
        columns.remove(label_col)
        self.x_columns = columns
        self.y_column = label_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        x, y = self.data.iloc[index][self.x_columns].values, \
               self.data.iloc[index][self.y_column]
        return x, int(y)


class ForestFiresCNNDataset(ForestFiresLinearDataset):

    def __getitem__(self, index) -> T_co:
        x, y = self.data.iloc[index][self.x_columns].values, \
               self.data.iloc[index][self.y_column]
        return x.reshape(1, -1), int(y)


class ForestFiresDataModule(LightningDataModule):

    def __init__(self, train_path: str, valid_path: str,
                 label_col: str, dataset_cls, batch_size: int = 32,
                 train_shuffle: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_path = train_path
        self.valid_path = valid_path
        self.batch_size = batch_size
        self.shuffle = train_shuffle
        self.label_col = label_col
        self.train_data = None
        self.valid_data = None
        self.generator = torch.Generator()
        self.generator.manual_seed(0)
        self.dataset_cls = dataset_cls
        self.load_data()

    def load_data(self):
        self.train_data = self.dataset_cls(self.train_path, self.label_col)
        self.valid_data = self.dataset_cls(self.valid_path, self.label_col)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            generator=self.generator, worker_init_fn=self.seed_worker
        )
        return loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        loader = DataLoader(
            self.valid_data, batch_size=self.batch_size, shuffle=False,
            generator=self.generator, worker_init_fn=self.seed_worker
        )
        return loader

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
