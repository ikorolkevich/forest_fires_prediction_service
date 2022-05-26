import os
from typing import NamedTuple

import pika
import dramatiq
from dramatiq.brokers.rabbitmq import RabbitmqBroker
from dramatiq.middleware import CurrentMessage


class RabbitmqParams(NamedTuple):
    username: str
    password: str
    host: str
    port: int
    vhost: str


class RedisParams(NamedTuple):
    host: str
    port: int
    db: str
    password: str


class TasksParams(NamedTuple):
    max_retries: int
    predict_forest_fire_queue: str
    get_weather_data_queue: str


RABBITMQ_PARAMS = RabbitmqParams(
    username=os.environ.get('RABBITMQ_USER_NAME', 'guest'),
    password=os.environ.get('RABBITMQ_USER_PASS', 'guest'),
    host=os.environ.get('RABBITMQ_HOST', '127.0.0.1'),
    port=int(os.environ.get('RABBITMQ_PORT', '5672')),
    vhost=os.environ.get('RABBITMQ_VHOST', 'vhost1'),
)


TASK_PARAMS = TasksParams(
    max_retries=int(os.environ.get('MAX_RETRIES', 0)),
    predict_forest_fire_queue=os.environ.get(
        'PREDICT_FOREST_FIRE_QUEUE', 'PREDICT_FOREST_FIRE'
    ),
    get_weather_data_queue=os.environ.get(
        'GET_WEATHER_DATA_QUEUE', 'GET_WEATHER_DATA_QUEUE'
    )
)


rabbitmq_broker = RabbitmqBroker(
    host=RABBITMQ_PARAMS.host, port=RABBITMQ_PARAMS.port,
    virtual_host=RABBITMQ_PARAMS.vhost,
    credentials=pika.PlainCredentials(
        username=RABBITMQ_PARAMS.username,
        password=RABBITMQ_PARAMS.password
    )
)
rabbitmq_broker.middleware = rabbitmq_broker.middleware[1:]
current_message_middleware = CurrentMessage()

for middleware in [
    current_message_middleware,
]:
    rabbitmq_broker.add_middleware(middleware)

dramatiq.set_broker(rabbitmq_broker)
