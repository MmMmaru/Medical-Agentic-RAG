import logging

# Initialize logger with basic configuration
logger = logging.getLogger("lightrag")
logger.propagate = False  # prevent log message send to root logger
logger.setLevel(logging.INFO)