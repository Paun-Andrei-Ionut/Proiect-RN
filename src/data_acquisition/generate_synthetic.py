import cv2
import numpy as np
import os
import random
from glob import glob

# --- CONFIGURARE ---
# Calea unde ai lăsat cele 225 de poze per clasă
INPUT_DIR = os.path.join("data", "raw") 

# Unde salvăm datele "originale"
OUTPUT_DIR = os.path.join("data", "generated")

# Generăm 150 pentru a atinge pragul de 40%
TARGET_PER_CLASS = 150 
CLASSES = ['glass', 'metal', 'paper', 'plastic']

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss * 50
    return np.clip(noisy, 0, 255).astype(np.uint8)

def adjust_brightness(image):
    value = random.randint(-50, 50)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        value = abs(value)
        lim = value
        v[v < lim] = 0
        v[v >= lim] -= value
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def generate_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print(f"🚀 Generăm date sintetice în: {OUTPUT_DIR}")

    for class_name in CLASSES:
        class_input_path = os.path.join(INPUT_DIR, class_name)
        class_output_path = os.path.join(OUTPUT_DIR, class_name)
        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)

        # Căutăm imaginile tale
        source_images = glob(os.path.join(class_input_path, "*.jpg")) + \
                        glob(os.path.join(class_input_path, "*.png")) + \
                        glob(os.path.join(class_input_path, "*.jpeg"))
        
        if not source_images:
            print(f"⚠️ Nu am găsit poze în {class_name}!")
            continue

        count = 0
        while count < TARGET_PER_CLASS:
            img_path = random.choice(source_images)
            img = cv2.imread(img_path)
            if img is None: continue

            transform_type = random.choice(['rotate', 'flip', 'noise', 'brightness', 'blur'])
            
            if transform_type == 'rotate':
                angle = random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
                aug_img = cv2.rotate(img, angle)
            elif transform_type == 'flip':
                aug_img = cv2.flip(img, random.choice([0, 1, -1]))
            elif transform_type == 'noise':
                aug_img = add_noise(img)
            elif transform_type == 'brightness':
                aug_img = adjust_brightness(img)
            elif transform_type == 'blur':
                aug_img = cv2.GaussianBlur(img, (5, 5), 0)

            save_path = os.path.join(class_output_path, f"synth_{class_name}_{count}.jpg")
            cv2.imwrite(save_path, aug_img)
            count += 1
            
        print(f"✅ Generat 150 imagini sintetice pentru {class_name}")

if __name__ == "__main__":
    generate_data()