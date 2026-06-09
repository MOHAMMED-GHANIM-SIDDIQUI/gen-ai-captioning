from pathlib import Path
import pickle
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
CAPTION_MODEL_PATH = MODEL_DIR / "model.keras"
FEATURE_EXTRACTOR_PATH = MODEL_DIR / "feature_extractor.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
MAX_LENGTH = 34
IMAGE_SIZE = 224


@st.cache_resource(show_spinner=False)
def load_assets():
    missing = [
        path.name
        for path in [CAPTION_MODEL_PATH, FEATURE_EXTRACTOR_PATH, TOKENIZER_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing model artifact(s): {', '.join(missing)}")

    caption_model = load_model(CAPTION_MODEL_PATH)
    feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)
    with TOKENIZER_PATH.open("rb") as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    return caption_model, tokenizer, feature_extractor


def extract_image_features(image_path: Path, feature_extractor):
    image = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return feature_extractor.predict(image_array, verbose=0)


def generate_caption(image_path: Path, model, tokenizer, feature_extractor) -> str:
    image_features = extract_image_features(image_path, feature_extractor)
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


def render_result(image_path: Path, caption: str) -> None:
    image = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(caption or "No caption generated", fontsize=14, color="navy")
        st.pyplot(fig)
    finally:
        plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="Image Caption Generator", layout="centered")
    st.title("Image Caption Generator")
    st.caption("Upload an image and generate a caption using the trained CNN-LSTM model.")

    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is None:
        st.info("Upload an image to begin.")
        return

    try:
        model, tokenizer, feature_extractor = load_assets()
        suffix = Path(uploaded_image.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpfile:
            tmpfile.write(uploaded_image.getbuffer())
            uploaded_image_path = Path(tmpfile.name)

        caption = generate_caption(uploaded_image_path, model, tokenizer, feature_extractor)
        st.success(caption or "Caption generation completed, but the model returned an empty caption.")
        render_result(uploaded_image_path, caption)
    except Exception as exc:
        st.error(f"Caption generation failed: {exc}")

    st.markdown("---")
    st.caption("Created by Mohammed Ghanim Siddiqui")


if __name__ == "__main__":
    main()
