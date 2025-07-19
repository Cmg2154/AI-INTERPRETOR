# 🤟 ASL Alphabet Recognition with TensorFlow + MobileNetV2

A deep learning model for recognizing American Sign Language (ASL) alphabets using a pre-trained MobileNetV2 architecture, built with TensorFlow and Keras.

---

## 📁 Dataset

This project uses the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).  
Make sure the dataset is extracted into the following structure:


Each folder contains thousands of 200x200 pixel images representing different ASL signs.

---

## ⚙️ Setup

### 1. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows

pip install tensorflow pillow matplotlib opencv-python

python run_inference.py
