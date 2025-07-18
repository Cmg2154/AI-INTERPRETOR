import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image

# ✅ 模型和标签路径
MODEL_PATH = 'checkpoints/epoch_03_val_acc_0.84.h5'
LABELS_PATH = 'data/asl_alphabet/asl_alphabet_train/asl_alphabet_train'

# ✅ 加载模型和标签
model = tf.keras.models.load_model(MODEL_PATH)
class_names = sorted(os.listdir(LABELS_PATH))

# ✅ 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("✅ 摄像头已开启，按 Q 退出程序")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取摄像头画面")
        break

    # 📌 定义 ROI（中间的正方形区域）
    h, w, _ = frame.shape
    roi_size = 300
    x1 = w // 2 - roi_size // 2
    y1 = h // 2 - roi_size // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size
    roi = frame[y1:y2, x1:x2]

    # ✅ 图像处理
    img = cv2.resize(roi, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # ✅ 模型预测
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = prediction[predicted_index]

    # ✅ 显示结果
    label_text = f"{predicted_label} ({confidence:.2f})"
    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 10), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()