import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Only add handler if logger doesn't already have one
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_logger = logging.StreamHandler()
        console_logger.setFormatter(formatter)
        logger.addHandler(console_logger)

    return logger
