import logging
import os
import sys


def my_custom_logger(logger_name, output_dir, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = (
        "%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
        "%(lineno)d — %(message)s"
    )
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(f"{output_dir}/{logger_name}", mode="a")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger
