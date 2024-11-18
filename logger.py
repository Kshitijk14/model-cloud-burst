import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """Custom formatter adding colors to the logs"""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[34;20m"
    reset = "\x1b[0m"
    
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logger(name='cloud_burst_prediction'):
    """
    Set up logger with both file and console handlers
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(CustomFormatter())
    
    # Create single consolidated log file handler
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = RotatingFileHandler(
        log_dir / f'consolidated_{current_time}.log',
        maxBytes=20*1024*1024,  # 20MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def setup_metrics_logger():
    """
    Set up a separate logger for metrics that writes to the same consolidated log file
    
    Returns:
        logging.Logger: Configured metrics logger instance
    """
    # Create metrics logger
    metrics_logger = logging.getLogger('metrics')
    metrics_logger.setLevel(logging.INFO)
    
    # Prevent adding handlers multiple times
    if metrics_logger.hasHandlers():
        metrics_logger.handlers.clear()
    
    # Get the existing file handler from the main logger
    main_logger = logging.getLogger('cloud_burst_prediction')
    for handler in main_logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            metrics_logger.addHandler(handler)
            break
    
    # Write CSV header to the consolidated log
    metrics_logger.info('METRICS_DATA: timestamp,data_shape,min_rain,max_rain,mean_rain,std_rain,prediction_min,prediction_max,prediction_mean')
    
    return metrics_logger

# Create global logger instances
logger = setup_logger()
metrics_logger = setup_metrics_logger()