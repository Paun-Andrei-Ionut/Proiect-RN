import os
import matplotlib.pyplot as plt
import tensorflow as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


# Detectăm automat căile
script_location = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_location)
data_dir = os.path.join(project_root, "data")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "validation")

# Parametrii rețelei (Hiperparametri)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Numărul de "ture" de învățare (poți crește la 20 dacă ai timp)
LEARNING_RATE = 0.001

def build_model(num_classes):
    print("Construction model CNN...")
    model = Sequential([
        Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        
        # Bloc 1 - Extragere trăsături simple (linii, colțuri)
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Bloc 2 - Extragere trăsături complexe (forme)
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Bloc 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Partea de Clasificare
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5), # Previne memorarea mecanică (Overfitting)
        Dense(num_classes, activation='softmax') # Stratul final (probabilități)
    ])
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # 1. Pregătirea Datelor (Augmentare + Încărcare)
    print("Se încarcă imaginile...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,      # Normalizare
        rotation_range=20,   # Augmentare: Rotire
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255) # Doar normalizare pt validare

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

    # 2. Construire Model
    num_classes = len(train_generator.class_indices)
    model = build_model(num_classes)
    model.summary() # Arată structura în consolă

    # 3. Antrenare
    print("\nÎNCEPE ANTRENAREA... (Poate dura câteva minute!)")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # 4. Salvare Rezultate
    # Salvăm modelul antrenat
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, "model_final.keras"))
    print(f"\nModel salvat în: models/model_final.keras")

    # 5. Generare Grafice (Acuratețe și Loss)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(EPOCHS)

    plt.figure(figsize=(12, 4))
    
    # Grafic Acuratețe
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Grafic Eroare (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    # Salvare grafic
    save_path = os.path.join(project_root, "docs", "images", "rezultate_antrenare.png")
    plt.savefig(save_path)
    print(f"Graficul rezultatelor salvat în: {save_path}")

if __name__ == "__main__":
    main()