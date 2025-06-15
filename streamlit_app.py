import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io

# Load the saved model once
@st.cache_resource
def load_classifier():
    return load_model('cat_dog_classifier.h5')

model = load_classifier()
IMG_SIZE = 128

def prepare_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

st.title('Cat vs Dog Classifier')
st.write('Upload an image of a cat or dog, and the model will predict which it is!')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write('')
    st.write('Classifying...')
    img = prepare_image(image)
    pred = model.predict(img)[0][0]
    label = 'Dog' if pred > 0.5 else 'Cat'
    confidence = pred if pred > 0.5 else 1 - pred
    confidence_percent = round(confidence * 100, 2)
    st.write(f'Prediction: **{label}**')
    st.write(f'Confidence: **{confidence_percent}%**') 