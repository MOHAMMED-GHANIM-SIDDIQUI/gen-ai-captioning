from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = APP_DIR / "models"
CAPTION_MODEL_PATH = MODEL_DIR / "model.keras"
FEATURE_EXTRACTOR_PATH = MODEL_DIR / "feature_extractor.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
MAX_LENGTH = 34
IMAGE_SIZE = 224
