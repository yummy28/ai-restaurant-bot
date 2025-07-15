import logging
import os

def setup_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

        os.makedirs('logs', exist_ok=True)

        file_handler = logging.FileHandler(f'logs/{log_file}')
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
