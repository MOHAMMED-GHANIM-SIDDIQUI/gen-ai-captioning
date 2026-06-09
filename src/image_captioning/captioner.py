from pathlib import Path
import pickle

import numpy as np

from .config import CAPTION_MODEL_PATH, FEATURE_EXTRACTOR_PATH, IMAGE_SIZE, MAX_LENGTH, TOKENIZER_PATH


def required_artifact_paths() -> list[Path]:
    return [CAPTION_MODEL_PATH, FEATURE_EXTRACTOR_PATH, TOKENIZER_PATH]


def missing_artifacts() -> list[str]:
    return [path.name for path in required_artifact_paths() if not path.exists()]


def assert_artifacts_available() -> None:
    missing = missing_artifacts()
    if missing:
        raise FileNotFoundError(f"Missing model artifact(s): {', '.join(missing)}")


def load_assets():
    from tensorflow.keras.models import load_model

    assert_artifacts_available()
    caption_model = load_model(CAPTION_MODEL_PATH)
    feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)
    with TOKENIZER_PATH.open("rb") as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    return caption_model, tokenizer, feature_extractor


def extract_image_features(image_path: Path, feature_extractor):
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    image = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return feature_extractor.predict(image_array, verbose=0)


def decode_caption(image_features, model, tokenizer) -> str:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    generated_text = "startseq"

    for _ in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([generated_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
        prediction = model.predict([image_features, sequence], verbose=0)
        word_index = int(np.argmax(prediction))
        word = tokenizer.index_word.get(word_index)
        if word is None:
            break
        generated_text += f" {word}"
        if word == "endseq":
            break

    return generated_text.replace("startseq", "").replace("endseq", "").strip()


def generate_caption(image_path: Path, model, tokenizer, feature_extractor) -> str:
    image_features = extract_image_features(image_path, feature_extractor)
    return decode_caption(image_features, model, tokenizer)
