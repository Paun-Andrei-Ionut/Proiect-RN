import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURARE ---
script_location = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_location))
train_dir = os.path.join(project_root, 'data', 'train')
val_dir = os.path.join(project_root, 'data', 'validation')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15 

# --- GENERATOARE ---
# Folosim preprocess_input care ajută enorm MobileNet-ul
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# --- MODEL ---
print("🚀 Descărcăm MobileNetV2...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Înghețăm baza

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(4, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("🔥 Începe antrenarea...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stop]
)

# --- SALVARE ---
models_dir = os.path.join(project_root, 'models')
if not os.path.exists(models_dir): os.makedirs(models_dir)
model.save(os.path.join(models_dir, 'model_final.keras'))
print("✅ Model salvat!")