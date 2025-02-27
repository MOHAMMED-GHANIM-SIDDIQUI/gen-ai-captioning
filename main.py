import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # Correct import for image processing
import matplotlib.pyplot as plt
import pickle
import os
import tempfile

# Function to generate and display caption
def generate_and_display_caption(image_path, model, tokenizer, feature_extractor, max_length=34, img_size=224):
    # Preprocess the image
    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)
    image_features = feature_extractor.predict(img, verbose=0)  # Extract image features

    # Generate the caption
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()

    # Display the image with the generated caption
    img = load_img(image_path, target_size=(img_size, img_size))
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption, fontsize=16, color='blue')
    st.pyplot(plt)  # Display image in Streamlit

# Streamlit app interface
def main():
    st.title("Image Caption Generator")
    st.write("Upload an image and generate a caption using the trained model.")

    # Upload the image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Save the uploaded image temporarily in the /tmp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
            tmpfile.write(uploaded_image.getbuffer())
            uploaded_image_path = tmpfile.name

        # Upload the model files (This can be optimized as well)
        model = load_model("models/model.keras")  # Make sure this path is correct for Streamlit
        tokenizer_path = "models/tokenizer.pkl"  # Ensure this path is correct or upload via Streamlit
        feature_extractor = load_model("models/feature_extractor.keras")  # Same for feature extractor

        # Load the tokenizer using pickle
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        # Generate caption and display image with caption
        generate_and_display_caption(uploaded_image_path, model, tokenizer, feature_extractor)
        
    st.markdown(" **Created by MOHAMMED GHANIM SIDIQUI** ")

if __name__ == "__main__":
    main()
