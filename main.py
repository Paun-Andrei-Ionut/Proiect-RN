import os
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- 1. CONFIGURARE ---
# Locația scriptului curent (src/app)
script_location = os.path.dirname(os.path.abspath(__file__))

# Urcăm 2 niveluri pentru a ajunge la rădăcina proiectului
# src/app -> src -> PROIECT_RETELE_NEURONALE
project_root = os.path.dirname(os.path.dirname(script_location))

# Calea către model (Verifică dacă ai .keras sau .h5)
# Dacă ai salvat ca trained_model.h5, schimbă numele aici!
model_path = os.path.join(project_root, "models", "model_final.keras")

# Etichetele claselor (OBLIGATORIU în ordine alfabetică, exact ca folderele)
# Am scos 'cardboard' și 'trash' cum ai cerut.
LABELS = ['glass', 'metal', 'paper', 'plastic']

# Stările sistemului (State Machine)
STATES = {
    "IDLE": 0,       # Așteaptă
    "CAPTURE": 1,    # Face poză
    "PREPROCESS": 2, # Pregătește imaginea
    "INFERENCE": 3,  # Gândește (AI)
    "DECISION": 4,   # Verifică siguranța
    "ACTUATION": 5,  # Arată rezultatul
    "ERROR": 99
}

class WasteSorterSystem:
    def __init__(self):
        self.state = STATES["IDLE"]
        self.model = None
        self.current_frame = None
        self.processed_image = None
        self.prediction = None
        self.confidence = 0.0
        
        print("[INIT] Sistemul pornește...")
        print(f"[DEBUG] Radacina detectata: {project_root}")
        self.load_resources()

    def load_resources(self):
        try:
            if not os.path.exists(model_path):
                print(f"\n[EROARE CRITICĂ] Nu găsesc modelul la: {model_path}")
                print("Verifică dacă numele fișierului din folderul 'models' este 'model_final.keras' sau 'trained_model.h5'")
                self.state = STATES["ERROR"]
                return

            print(f"[INIT] Încărcare model din: {model_path} ...")
            print("(Acest pas poate dura 10-20 secunde, te rog așteaptă!)")
            
            self.model = load_model(model_path)
            
            # Verificăm dacă modelul așteaptă 4 clase sau 6
            output_shape = self.model.output_shape
            print(f"[DEBUG] Modelul așteaptă {output_shape[-1]} clase.")
            
            if output_shape[-1] != len(LABELS):
                print(f"[ATENȚIE] Modelul a fost antrenat pe {output_shape[-1]} clase, dar tu ai definit {len(LABELS)}.")
                print("Dacă primești erori, verifică lista LABELS din cod.")

            print("[INIT] Model încărcat cu succes!")
        except Exception as e:
            print(f"[EROARE] Nu s-a putut încărca modelul: {e}")
            self.state = STATES["ERROR"]

    def run(self):
        if self.state == STATES["ERROR"]:
            print("Sistemul este în eroare. Ieșire.")
            return

        # Deschidem camera (0 = webcam default)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[EROARE] Nu pot deschide camera web!")
            return

        print("\n--- SISTEM PREGĂTIT ---")
        print("1. Pune obiectul în chenarul galben.")
        print("2. Apasă 'SPACE' pentru a detecta.")
        print("3. Apasă 'Q' pentru a ieși.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Eroare la citirea camerei.")
                break

            # Imaginea principală pe care desenăm textul
            display_frame = frame.copy()
            
            # Desenăm un pătrat în centru (ROI)
            h, w, _ = frame.shape
            box_size = 224 # Dimensiunea intrării în rețea
            start_x = (w - box_size) // 2
            start_y = (h - box_size) // 2
            end_x = start_x + box_size
            end_y = start_y + box_size
            
            # Chenar vizual pentru utilizator
            cv2.rectangle(display_frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)

            # Afișăm starea curentă
            state_name = [k for k, v in STATES.items() if v == self.state][0]
            cv2.putText(display_frame, f"STATE: {state_name}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- LOGICA STATE MACHINE ---
            
            # 1. IDLE - Așteaptă comanda
            if self.state == STATES["IDLE"]:
                cv2.putText(display_frame, "Apasă SPACE pentru detectie", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 2. CAPTURE - Face "poza"
            elif self.state == STATES["CAPTURE"]:
                print("[STATE] Capture...")
                self.current_frame = frame.copy()
                self.state = STATES["PREPROCESS"]

            # 3. PREPROCESS - Decupează și pregătește pt AI
            elif self.state == STATES["PREPROCESS"]:
                # Decupăm exact pătratul din centru
                cropped_img = self.current_frame[start_y:end_y, start_x:end_x]
                
                # Afișăm mic ce vede AI-ul (Debug)
                cv2.imshow("Input Retea", cropped_img)

                # Resize și Normalizare
                img = cv2.resize(cropped_img, (224, 224))
                img = img.astype("float32") / 255.0
                self.processed_image = np.expand_dims(img, axis=0)
                
                self.state = STATES["INFERENCE"]

            # 4. INFERENCE - Rețeaua gândește
            elif self.state == STATES["INFERENCE"]:
                try:
                    preds = self.model.predict(self.processed_image, verbose=0)
                    idx = np.argmax(preds)
                    
                    # Verificare limite array (dacă modelul prezice o clasă care nu e în listă)
                    if idx < len(LABELS):
                        self.prediction = LABELS[idx]
                        self.confidence = float(np.max(preds))
                    else:
                        self.prediction = "UNKNOWN"
                        self.confidence = 0.0

                    print(f"[AI] Predicție: {self.prediction} ({self.confidence*100:.2f}%)")
                    self.state = STATES["DECISION"]
                except Exception as e:
                    print(f"[EROARE INFERENCE] {e}")
                    self.state = STATES["IDLE"]

            # 5. DECISION - E destul de sigur?
            elif self.state == STATES["DECISION"]:
                THRESHOLD = 0.50  # 50% siguranță minimă
                
                if self.confidence < THRESHOLD:
                    print(f"[DECISION] Nesigur ({self.confidence*100:.1f}%). Respins.")
                    self.prediction = "UNKNOWN"
                
                self.state = STATES["ACTUATION"]

            # 6. ACTUATION - Arată rezultatul final
            elif self.state == STATES["ACTUATION"]:
                # Culori: Roșu (Unknown) vs Verde (Ok)
                color = (0, 0, 255) if self.prediction == "UNKNOWN" else (0, 255, 0)
                
                # Text mare pe ecran
                msg = f"{self.prediction.upper()} ({self.confidence*100:.0f}%)"
                
                # Centrare text (aproximativ)
                text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(display_frame, msg, (text_x, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                cv2.imshow('Garbage Sorter AI', display_frame)
                
                # Așteaptă 3 secunde să vezi rezultatul, apoi resetează
                print("[ACTUATION] Afișare rezultat...")
                cv2.waitKey(3000) 
                
                self.state = STATES["IDLE"]

            # --- AFIȘARE ȘI INPUT ---
            cv2.imshow('Garbage Sorter AI', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and self.state == STATES["IDLE"]:
                self.state = STATES["CAPTURE"]

        # La final
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = WasteSorterSystem()
    app.run()