import logging
import sys
import os

log_dir = '/mnt/data/sami/logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/mnt/data/sami/logs/game_data_pipeline.log')
    ]
)