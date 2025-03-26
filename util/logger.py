import logging
from datetime import datetime
from pytz import timezone
import os
import sys


def get_logger(log_path, name):
    
    logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()
    # log
    os.makedirs(log_path, exist_ok=True)
    file_handler = logging.FileHandler(filename=f"{log_path}/{name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    date = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)
    logging.info("Start logging " + name)
    
    return logging
