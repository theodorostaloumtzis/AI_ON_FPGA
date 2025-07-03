import logging

def get_logger(name: str = "ai_on_fpga") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:            # avoid dupes on reload
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

log = get_logger()
