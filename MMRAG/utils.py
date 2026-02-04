import logging
import os
from typing import Optional
import re

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
	pass
def format_references():
	pass
def compute_mdhash_id(content: str, prefix: str) -> str:
	import hashlib
	mdhash = hashlib.md5(content.encode('utf-8')).hexdigest()
	mdhash_id = f"{prefix}{mdhash}"
	return mdhash_id


def extract_catogorical_answer(text: str) -> str:
    """Return A/B/C/D if found in model output, else ''."""
    content_match = re.search(r"<answer>(.*?)</answer>", text)
    given_answer = content_match.group(1).strip() if content_match else text.strip()
    final_answer = re.search(r"\b([A-E])\b", given_answer)
    return final_answer.group(1) if final_answer else ""


def encode_image_paths_to_base64(image_paths: list[str], content_format = "image_url"):
	"""
	encode image paths to base64 strings
	format: [
			'type': 'image_url', 
		'image_url': {'url': image_content},
	]
	"""
	import base64
	import requests
	import os
	from PIL import Image
	import io

	content = []
	for image in image_paths:
		image_content = None
		if isinstance(image, str):
			if image.startswith(('http', 'https', 'oss')):
				response = requests.get(image)
				image_str = base64.b64encode(response.content).decode("utf-8")
				image_content = f"data:image/jpeg;base64,{image_str}"
			elif os.path.exists(image):
				abs_image_path = os.path.abspath(image)
				mime_type = "image/png" if abs_image_path.lower().endswith(".png") else "image/jpeg"
				with open(abs_image_path, "rb") as img_file:
					img_str = base64.b64encode(img_file.read()).decode("utf-8")
				image_content = f"data:{mime_type};base64,{img_str}"
			elif isinstance(image, Image.Image):
				buffered = io.BytesIO()
				image.save(buffered, format="JPEG")
				img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
				image_content = f"data:image/jpeg;base64,{img_str}"
		else:
			raise ValueError("Unsupported image format")
		
		if image_content:
			if content_format == "image_url":
				content.append({
					'type': 'image_url', 
					'image_url': {'url': image_content},
				})
			else:
				content.append({
					'type': 'image', 
					'image': image_content,
				})
	return content
if __name__ == "__main__":
	logger.info("This is a test log message.")