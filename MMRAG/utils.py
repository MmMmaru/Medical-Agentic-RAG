import logging
import os
from typing import Optional

def get_logger(name: str = "MMRAG", level: Optional[str] = None, log_file: Optional[str] = None) -> logging.Logger:
	"""
	Create or return a configured logger.

	Env vars:
	  - MMRAG_LOG_LEVEL: DEBUG/INFO/WARNING/ERROR
	  - MMRAG_LOG_FILE: path to log file (optional)
	"""
	logger = logging.getLogger(name)
	if logger.handlers:
		return logger

	resolved_level = (level or os.getenv("MMRAG_LOG_LEVEL", "INFO")).upper()
	logger.setLevel(resolved_level)

	formatter = logging.Formatter(
		fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

	file_path = log_file or os.getenv("MMRAG_LOG_FILE")
	if file_path:
		file_handler = logging.FileHandler(file_path)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

	logger.propagate = False
	return logger

logger = get_logger()

def build_context():

def format_references():


if __name__ == "__main__":
	logger.info("This is a test log message.")