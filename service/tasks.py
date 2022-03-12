import dramatiq

import socketio

from service.decorators import done_for
from service.dramatiqconfig import TASK_PARAMS


sio = socketio.Client()
sio.connect(
    "http://localhost:8000/ws", namespaces="/nd_prediction", socketio_path="/ws/socket.io"
)


@sio.on("message", namespace="/nd_prediction")
def new_packet(packet):
    print("Message: ", packet)


def call_back(data):
    print("call-back", data)


@dramatiq.actor(
    max_retries=TASK_PARAMS.max_retries,
    queue_name=TASK_PARAMS.predict_forest_fire_queue
)
@done_for
def predict(msg: str, key: str) -> None:
    sio.emit(
        "packet",
        {'content': {"message": msg, 'sid': key}, 'content_type': "application/json"},
        namespace="/nd_prediction"
    )
