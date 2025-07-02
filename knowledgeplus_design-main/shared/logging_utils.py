"""Logging utilities shared by the Streamlit apps."""

import logging
from logging import FileHandler, StreamHandler, Formatter


def configure_logging(log_file: str = "rag_tool.log", level: int = logging.INFO) -> None:
    """Configure the root logger once with uniform handlers.

    Parameters
    ----------
    log_file : str
        Path to the file where logs will be written.
    level : int
        Logging level for the root logger.
    """

    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level)
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = Formatter(fmt)

    file_handler = FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)
