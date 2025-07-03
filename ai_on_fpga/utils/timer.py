import time
from contextlib import contextmanager
from .log import log

@contextmanager
def timer(msg: str = "elapsed"):
    start = time.time()
    yield
    log.info("%s: %.2f s", msg, time.time() - start)
