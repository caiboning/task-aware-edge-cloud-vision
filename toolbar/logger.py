
import logging

from toolbar import config


# ==============================
# Logger Configuration
# ==============================

def get_logger(name="pipeline"):
    """
    Create or reuse a logger.

    INFO:
        Normal execution

    WARNING:
        Unexpected but recoverable events

    ERROR:
        Processing failures
    """

    config.PIPELINE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --------------------------
    # File Handler
    # --------------------------

    file_handler = logging.FileHandler(
        config.PIPELINE_LOG,
        mode="a",
        encoding="utf-8"
    )

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # --------------------------
    # Console Handler
    # --------------------------

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
