import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

# ✅ 检查 GPU 是否可用
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU 可用，使用 GPU 进行训练")
else:
    print("⚠️ 没有检测到 GPU，使用 CPU 进行训练")

# ✅ 自动创建 checkpoints 文件夹
os.makedirs('checkpoints', exist_ok=True)

# ✅ 数据路径（你提供的）
train_dir = os.path.join("data", "asl_alphabet", "asl_alphabet_train", "asl_alphabet_train")

# ✅ 图像增强 & 数据生成器
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% 用作验证集
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ✅ 使用 MobileNetV2 作为预训练模型
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False  # 冻结预训练权重

# ✅ 构建模型
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ 设置 checkpoint 回调函数
checkpoint_callback = ModelCheckpoint(
    filepath='checkpoints/epoch_{epoch:02d}_val_acc_{val_accuracy:.2f}.h5',
    save_best_only=False,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# ✅ 模型训练
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[checkpoint_callback]
)