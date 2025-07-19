# 🐾 AI-Powered Animal Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-red)

A deep learning-based web app that predicts the animal species from an uploaded image using a fine-tuned MobileNetV2 model. Built with **Streamlit**, this classifier supports real-time predictions across **19+ animal categories**.

---

## 🚀 Features

- 🐕 Upload animal image (JPG, PNG, JPEG, WEBP)
- 🔍 Predicts among 19+ animal species
- 💬 Popup result with prediction label
- 🔁 One-click refresh button
- 📱 Responsive and user-friendly UI
- ⚙️ Powered by MobileNetV2 + Streamlit

---

## 📸 App Preview

| Upload Interface | Prediction Popup |
|------------------|------------------|
| ![Upload Interface](images/1.png) | ![Popup Result](images/2.png) |
| ![Another View](images/3.png) | 

---

## 🧠 Model Info

- 🧪 **Architecture:** MobileNetV2 (Transfer Learning)
- 🖼️ **Input Size:** 224 x 224 pixels
- 🧩 **Classes:** 19 animal types
- 🔢 **Batch Size:** 16
- 🔁 **Epochs:** 25
- 🎯 **Accuracy:** ~92% on test set
- 🧠 **Framework:** TensorFlow & Keras

---

## 🛠️ Tech Stack

- Python 3.8+
- TensorFlow / Keras
- MobileNetV2 (pre-trained)
- Streamlit
- NumPy
- Pillow (PIL)

---

## 📁 Project Structure

AI-Powered-Animal-Classifier/
├── app.py
├── model/
│ └── animal_model.h5
├── dataset/
├── images/
│ ├── 1.png
│ ├── 2.png
│ ├── 3.png
│ └── 4.png
├── requirements.txt
└── README.md


---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/DevWaqarAhmad/AI-Powered-Animal-Classifier.git
cd AI-Powered-Animal-Classifier

pip install -r requirements.txt
streamlit run app.py

🤖 How It Works
Upload an image of an animal.

The model resizes and preprocesses the image.

MobileNetV2 predicts the most likely animal class.

The prediction is shown in a popup-style card with label and confidence.

👨‍💻 Author
Waqar Ahmad
📧 devwaqarahmad@gmail.com