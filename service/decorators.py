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
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.error(
                f'Error in {fn.__name__.upper()} {message.message_id}: {e}'
            )
        logger.info(f'{fn.__name__.upper()} {message.message_id} done for '
                    f'{datetime.datetime.now()-begin_at}')
        return
    return wrapper
