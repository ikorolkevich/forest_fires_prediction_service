import os
from dataclasses import dataclass


@dataclass
class DatabaseParams:
    dbname: str
    user: str
    password: str
    host: str
    port: int


DB_PARAMS = DatabaseParams(
    dbname=os.environ.get('DB_NAME', 'db'),
    user=os.environ.get('DB_USER', 'user'),
    password=os.environ.get('DB_PASSWORD', 'password'),
    host=os.environ.get('DB_HOST', 'localhost'),
    port=int(os.environ.get('DB_PORT', '5432'))
)


WEATHER_API_KEY = os.environ.get(
    'WEATHER_API_KEY', 'acc6e9b0a74d1f00ad31679aa42e7ed7'
)

MODEL_PATH = '/home/ikorolkevich/git/forest_fires_prediction_service/' \
             'forest_fires_classifier/test/' \
             'Скорость обучения 0.0001 Размер партии 32/version_0/' \
             'checkpoints/model-epoch=19-val_f1score=0.91174.ckpt'
