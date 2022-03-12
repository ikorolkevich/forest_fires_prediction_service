import itertools
import random

import numpy as np
import torch
from pytorch_lightning import callbacks, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from forest_fires_classifier.classifier import ForestFireClassifier, \
    ForestFiresClassifierCNN, ForestFireClassifier64, ForestFireClassifier128Batch
from forest_fires_classifier.dataset import ForestFiresDataModule, \
    ForestFiresLinearDataset, ForestFiresCNNDataset

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def run_train(
        batch_size, learning_rate, func_act,
        train_file='/home/ikorolkevich/git/forest_fires_prediction_service/'
                   'forest_fires_classifier/data_processing/train.csv',
        valid_file='/home/ikorolkevich/git/forest_fires_prediction_service/'
                   'forest_fires_classifier/data_processing/test.csv',
        output_dir='grid-logs-linear128batch',
        epochs=50,
        patience=10,
):
    dataset = ForestFiresDataModule(
        train_path=train_file, valid_path=valid_file,
        batch_size=batch_size, label_col='label',
        dataset_cls=ForestFiresLinearDataset
    )

    # model = ForestFireClassifier(lr=learning_rate, num_classes=5, func_act=func_act)
    # model = ForestFiresClassifierCNN(lr=learning_rate, num_classes=5, func_act=func_act)
    model = ForestFireClassifier128Batch(lr=learning_rate, num_classes=5, func_act=func_act)
    model_name = f'{model.__class__.__name__}_act_' \
                 f'{model.func_act.__class__.__name__}_lr_' \
                 f'{learning_rate}_bs_{batch_size}'
    print(f'Start training {model_name}')
    tb_logger = TensorBoardLogger(
        save_dir=output_dir, name=model_name
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
        patience=patience,
        verbose=True,
        mode='max',
        check_finite=True,
        stopping_threshold=0.99999
    )
    trainer = Trainer(
        gpus=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tb_logger, max_epochs=epochs
    )
    trainer.fit(model, datamodule=dataset)


if __name__ == '__main__':
    PARAMETERS = {
        'learning_rate': [1e-3, 3e-3, 1e-2, 3e-2, 1e-4, 3e-4],
        'batch_size': [16, 32, 64],
        # 'func_act': [nn.ReLU, nn.GELU, nn.Mish, nn.ELU, nn.LeakyReLU, nn.SELU],
        'func_act': [nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Mish, nn.ELU, nn.SELU],
    }
    keys, values = zip(*PARAMETERS.items())
    permutations_dicts = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]
    for params in permutations_dicts:
        run_train(**params)

