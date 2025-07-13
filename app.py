import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

model = tf.keras.models.load_model('model/animal_model.h5')

# Class labels (same order as during training)
class_labels = list(model.classes) if hasattr(model, 'classes') else [
    'Crocodile', 'camel', 'cat', 'chicken', 'cow', 'deer', 'dog', 'donkey',
    'elephant', 'fishe', 'foxe', 'giraffe', 'goat', 'horse', 'kangaro',
    'leopard', 'lion', 'monkey', 'panda'
]

st.title("🐾 Animal Image Classifier")
st.write("Upload an image and let the model predict the animal!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    st.success(f"Prediction: **{predicted_class}** 🐾")

