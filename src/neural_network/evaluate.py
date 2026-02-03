import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURARE ---
script_location = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_location))
test_dir = os.path.join(project_root, 'data', 'test')
model_path = os.path.join(project_root, 'models', 'model_final.keras')

# REPARAT: MobileNet cere exact 224x224
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def evaluate_model():
    # 1. Verificări
    if not os.path.exists(model_path):
        print(f"EROARE: Nu găsesc modelul la {model_path}")
        return
    
    if not os.path.exists(test_dir):
        print(f"EROARE: Nu găsesc folderul de test la {test_dir}")
        return

    print("⏳ Încărcăm modelul...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Eroare la încărcarea modelului: {e}")
        return

    # 2. Generator Date Test 
    # CRUCIAL: Folosim preprocess_input pentru MobileNet
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    print(f"📂 Încărcăm imagini din: {test_dir}")
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE, # Acum e 224x224
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False # IMPORTANT: Nu amestecăm, ca să comparăm corect
    )

    # 3. Predicție
    print("Calculăm scorul general...")
    results = model.evaluate(test_generator)
    print(f"\n✅ REZULTAT FINAL PE TEST:\nAcuratețe: {results[1]*100:.2f}%")

    # 4. Matrice de Confuzie
    print("\nGenerăm raportul detaliat...")
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Raport Text
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # Grafic
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicția Modelului')
    plt.ylabel('Realitate')
    plt.title(f'Matrice Confuzie (Acuratețe: {results[1]*100:.1f}%)')
    
    save_path = os.path.join(project_root, 'docs', 'images', 'confusion_matrix.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nGraficul a fost salvat în: {save_path}")
    print("Poți închide fereastra cu graficul pentru a finaliza.")
    plt.show()

if __name__ == "__main__":
    evaluate_model()