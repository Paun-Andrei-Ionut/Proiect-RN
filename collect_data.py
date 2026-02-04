import cv2
import os
import time
import numpy as np

# --- CONFIGURARE ---
# Mergem 2 niveluri mai sus ca să ajungem la folderul principal (din src/data_acquisition -> src -> root)
script_location = os.path.dirname(os.path.abspath(__file__))
# Urcăm de două ori (dirname) pentru a ieși din 'data_acquisition' și din 'src'
project_root = os.path.dirname(os.path.dirname(script_location))

# Salvăm în data/generated (ca să se vadă că sunt datele TALE)
save_dir = os.path.join(project_root, "data", "generated")

# Asigurăm structura de foldere
categories = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
for cat in categories:
    os.makedirs(os.path.join(save_dir, cat), exist_ok=True)

def collect_images():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Eroare: Nu pot deschide camera!")
        return

    print("\n--- DATA COLLECTOR (Modul Achiziție) ---")
    print("Pune obiectul în chenar și apasă tasta:")
    print(" [g] - Glass (Sticlă)")
    print(" [p] - Paper (Hârtie)")
    print(" [c] - Cardboard (Carton)")
    print(" [l] - Plastic")
    print(" [m] - Metal")
    print(" [t] - Trash (Altele)")
    print(" [q] - IEȘIRE\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Desenăm chenarul (ca să știi ce pozezi)
        h, w, _ = frame.shape
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        end_x = start_x + min_dim
        end_y = start_y + min_dim
        
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(display_frame, "DATA COLLECTOR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Acquisiton Module", display_frame)
        
        # 2. Ascultăm tastele
        key = cv2.waitKey(1) & 0xFF
        category = None
        
        if key == ord('q'): break
        elif key == ord('g'): category = 'glass'
        elif key == ord('p'): category = 'paper'
        elif key == ord('c'): category = 'cardboard'
        elif key == ord('l'): category = 'plastic'
        elif key == ord('m'): category = 'metal'
        elif key == ord('t'): category = 'trash'
        
        # 3. Dacă ai apăsat o tastă validă, salvăm poza
        if category:
            # Decupăm doar pătratul (ROI)
            roi = frame[start_y:end_y, start_x:end_x]
            # Facem resize la 224x224 (standardul proiectului)
            roi = cv2.resize(roi, (224, 224))
            
            # Generăm nume unic
            timestamp = int(time.time() * 1000)
            filename = f"{category}_{timestamp}.jpg"
            save_path = os.path.join(save_dir, category, filename)
            
            cv2.imwrite(save_path, roi)
            print(f"✅ [SALVAT] {category}: {filename}")
            
            # Flash vizual (ecran alb scurt)
            cv2.imshow("Acquisiton Module", np.ones_like(display_frame)*255)
            cv2.waitKey(50)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_images()