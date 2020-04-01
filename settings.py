import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

TINYIMAGES_DATA_PATH = os.getenv("TINYIMAGES_DATA_PATH")
TINYIMAGES_META_PATH = os.getenv("TINYIMAGES_META_PATH")
