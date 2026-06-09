from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img

from .captioner import generate_caption, load_assets
from .config import IMAGE_SIZE


@st.cache_resource(show_spinner=False)
def cached_assets():
    return load_assets()


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
        model, tokenizer, feature_extractor = cached_assets()
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
