import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Animal Classifier", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        text-align: center;
        font-size: 28px;
        color: #4a4a4a;
        margin-bottom: 0px;
    }
    .btn-refresh {
        display: block;
        margin: auto;
        background-color: #1373F9;
        color: white;
        padding: 10px 25px;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        font-size: 16px;
        transition: background-color 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .btn-refresh:hover {
        background-color: #172661;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main"><div class="title">🐾 AI Animal Classifier App 🐾</div>', unsafe_allow_html=True)

# Refresh button
st.markdown("""
<a href="/" target="_self">
    <button class="btn-refresh">🔄 Refresh</button>
</a>
""", unsafe_allow_html=True)

# Load model
model_path = "model/animal_model.h5"
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ['Crocodiles', 'camels', 'cats', 'chickens', 'cows', 'deers', 'dogs',
                'donkeys', 'elephants', 'fishes', 'foxes', 'giraffes', 'goats',
                'horses', 'kangaros', 'leopards', 'lions', 'monkeys', 'pandas']

# Upload
uploaded_file = st.file_uploader("📤 Upload an animal image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    # Popup using JS + HTML injected via iframe
    popup_html = f"""
    <div id="popup" style="
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: #e3f2fd;
        padding: 30px 40px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        z-index: 9999;
        text-align: center;
        font-size: 20px;
        font-weight: 500;
        color: #0d47a1;
        border: 2px solid #2196f3;
    ">
        🧠 <strong>Model Prediction</strong><br><br>
        This image is most likely a <strong style="color:#1b5e20;">{predicted_class}</strong> 🐾
        <br><br>
        <button onclick="document.getElementById('popup').style.display='none'" style="
            background-color: #f44336;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
        ">Close</button>
    </div>
    """

    st.components.v1.html(popup_html, height=300)

# Close container
st.markdown("</div>", unsafe_allow_html=True)
