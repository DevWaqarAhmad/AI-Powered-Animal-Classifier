import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="AI Animal Classifier", layout="centered")

# Initialize session state to handle popup visibility
if "show_popup" not in st.session_state:
    st.session_state.show_popup = False

# --- Custom CSS ---
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
        display: inline-block;
        background-color: #1373F9;
        color: white;
        padding: 10px 25px;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        font-size: 16px;
        transition: background-color 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        text-decoration: none;
        margin-top: 15px;
    }
    .btn-refresh:hover {
        background-color: #172661;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown('<div class="main"><div class="title">🐾 AI Animal Classifier App 🐾</div>', unsafe_allow_html=True)

# --- Refresh Button (normal, optional) ---
# st.markdown("""
# <a href="/" target="_self">
#     <button class="btn-refresh">🔄 Refresh</button>
# </a>
# """, unsafe_allow_html=True)

# --- Load Model ---
model_path = "model/animal_model.h5"
model = tf.keras.models.load_model(model_path)

# --- Class Labels ---
class_labels = ['Crocodile', 'Camel', 'Cat', 'Chicken', 'Cow', 'Deer', 'Dog',
                'Donkey', 'Elephant', 'Fish', 'Foxe', 'Giraffe', 'Goat',
                'Horse', 'Kangaro', 'Leopard', 'Lion', 'Monkey', 'Panda']

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload an animal image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    st.session_state.show_popup = True

# --- Display Popup ---
if st.session_state.show_popup:
    st.markdown(f"""
        <div style="
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
            This image is most likely a <strong style='color:#1b5e20;'>{predicted_class}</strong> 🐾
            <br><br>
            <a href='/' target='_self' class='btn-refresh' style='color: white;'>🔄 Refresh</a>
            </div>
    """, unsafe_allow_html=True)

# --- Close Main Div ---
st.markdown("</div>", unsafe_allow_html=True)
