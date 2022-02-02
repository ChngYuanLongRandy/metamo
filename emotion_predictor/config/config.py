import os
from pathlib import Path

# main folders
CONFIG_PATH =os.path.dirname(__file__)  # emotion_predictor/config
ROOT = Path(CONFIG_PATH).parent  # emotion_predictor
DATA_PATH = os.path.join(ROOT, 'data')
MODEL_PATH = os.path.join(ROOT, 'model')

# file paths

