import datetime
import logging
from functools import wraps

from dramatiq.middleware import CurrentMessage

logger = logging.getLogger(__name__)


def done_for(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        message = CurrentMessage.get_current_message()
        begin_at = datetime.datetime.now()
        result = fn(*args, **kwargs)
        logger.info(f'{fn.__name__.upper()} {message.message_id} done for '
                    f'{datetime.datetime.now()-begin_at}')
        return result
    return wrapper
