import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os

# Constants
IMG_SIZE = 128
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "cat_dog_classifier.h5"

# Load model only once
@st.cache_resource
def load_classifier():
    return load_model(MODEL_PATH)

model = load_classifier()

# Prepare image for prediction
def prepare_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Streamlit UI
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

st.title("🐱🐶 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an image of a Cat or Dog", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save image to disk
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Predict
    image = Image.open(uploaded_file).convert("RGB")
    processed_image = prepare_image(image)
    prediction = model.predict(processed_image)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    confidence_percent = round(confidence * 100, 2)

    # Result
    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"### Confidence: **{confidence_percent}%**")
