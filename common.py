import logging


class LoopRetryError(Exception):
    def __init__(self):
        ...


def retry_with_log(error):
    if not isinstance(error, LoopRetryError):
        logging.warning(error)
    return True


def init_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
