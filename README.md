# 🤟 ASL Alphabet Recognition with TensorFlow + MobileNetV2

A deep learning model for recognizing American Sign Language (ASL) alphabets using a pre-trained MobileNetV2 architecture, built with TensorFlow and Keras.

---

## 📁 Dataset

This project uses the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).  
Download dataset as zip (1GB)
After downloading, make sure the dataset is extracted and placed like this:
data/asl_alphabet/asl_alphabet_train/asl_alphabet_train 




## ⚙️ Setup

### Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows

pip install tensorflow pillow matplotlib opencv-python

python run_inference.py


##
