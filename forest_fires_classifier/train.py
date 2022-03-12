import argparse
import random

import numpy as np
import torch
from pytorch_lightning import callbacks, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from forest_fires_classifier.classifier import ForestFireClassifier
from forest_fires_classifier.dataset import ForestFiresDataModule, \
    ForestFiresLinearDataset

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--train_file', type=str, default='data_processing/train.csv'
    )
    parser.add_argument(
        '-v', '--valid_file', type=str, default='data_processing/test.csv'
    )
    parser.add_argument('-lr', '--learning_rate', type=float, default=7e-5)
    parser.add_argument('-m', '--model_name', type=str, default='model_linear_leaky_relu_7e-5')
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('-p', '--patience', type=int, default=10)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-o', '--output_dir', type=str, default='train-logs')
    args = parser.parse_args()

    dataset = ForestFiresDataModule(
        train_path=args.train_file, valid_path=args.valid_file,
        batch_size=args.batch_size, label_col='label',
        dataset_cls=ForestFiresLinearDataset
    )

    model = ForestFireClassifier(lr=args.learning_rate, num_classes=5)

    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir, name=args.model_name
    )
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor='val_f1score',
        filename='model-{epoch:02d}-{val_f1score:.5f}',
        save_top_k=2,
        mode='max',
        save_last=True,
        verbose=True,
    )
    early_stopping_callback = callbacks.EarlyStopping(
        monitor='val_f1score',
        min_delta=0.001,
        patience=args.patience,
        verbose=True,
        mode='max',
        check_finite=True,
        stopping_threshold=0.99999
    )
    trainer = Trainer(
        gpus=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tb_logger, max_epochs=args.epochs
    )
    trainer.fit(model, datamodule=dataset)

    grid_search = {
        'act': ['relu', 'gelu', 'Mish', "ELU", "LeakyReLU", 'selu'],
        'layers': 'normalization before act',
        'lrs': [1e-3, 3e-3, 1e-2, 3e-2, 1e-4, 3e-4],
    }