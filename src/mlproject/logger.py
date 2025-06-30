"""
Logger configuration for the stock portfolio project.
Sets up logging to a timestamped file in the logs directory.
"""
import logging
import os
from datetime import datetime

logs_dir = os.path.join(r"C:\Users\katch\Desktop\projects\stock_portfolio", "logs")
os.makedirs(logs_dir, exist_ok=True)

log_file = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
log_file_path = os.path.join(logs_dir, log_file)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='[%(asctime)s]  %(lineno)d %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
