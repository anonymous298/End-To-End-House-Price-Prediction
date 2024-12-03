import os
import logging
from datetime import datetime

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=PATH,
    filemode='a',
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name (str): The name of the logger (usually the module or file name).
    
    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)
    return logger