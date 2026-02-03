import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # IMPORT IMPORTANT
import os

# --- CONFIGURARE PAGINĂ ---
st.set_page_config(
    page_title="SIA Reciclare - Etapa 6",
    page_icon="♻️",
    layout="wide"
)

# --- CĂI ȘI CLASE ---
# Ne asigurăm că calea este relativă la folderul de unde rulăm
script_location = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_location))
MODEL_PATH = os.path.join(project_root, 'models', 'model_final.keras')

CLASSES = ['glass', 'metal', 'paper', 'plastic']

# --- FUNCȚIE PENTRU LOGICA CELOR 4 COȘURI ---
def get_bin_info(label):
    if label == 'paper':
        return {
            "bin_name": "COȘUL ALBASTRU (Hârtie/Carton)",
            "color": "blue",
            "icon": "🟦",
            "msg": "Hârtia trebuie să fie curată și uscată."
        }
    elif label == 'plastic':
        return {
            "bin_name": "COȘUL GALBEN (Plastic)",
            "color": "gold",
            "icon": "🟨",
            "msg": "Turtiți PET-urile! Spălați recipientele."
        }
    elif label == 'glass':
        return {
            "bin_name": "COȘUL VERDE (Sticlă)",
            "color": "green",
            "icon": "🟩",
            "msg": "Fără capace. Sticla se spală înainte."
        }
    elif label == 'metal':
        return {
            "bin_name": "COȘUL ROȘU / GRI (Metal)",
            "color": "red",
            "icon": "🟥",
            "msg": "Doze de aluminiu, conserve. Vă rugăm să le clătiți."
        }
    return None

# --- ÎNCĂRCARE MODEL (CACHED) ---
@st.cache_resource
def load_app_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Eroare la încărcarea modelului: {e}")
        return None

# --- INTERFAȚA PRINCIPALĂ ---
st.title("♻️ Sistem Inteligent de Sortare Deșeuri")
st.markdown("**Status:** ✅ Model MobileNetV2 Activat | **Acuratețe:** 75.44%")

# Sidebar
st.sidebar.title("Meniu Control")
st.sidebar.info("Acest modul software simulează stația de sortare industrială.")
app_mode = st.sidebar.selectbox("Sursă Imagine:", ["📸 Camera Live", "📂 Încărcare Fișier"])

model = load_app_model()

if model is None:
    st.error(f"⚠️ EROARE CRITICĂ: Nu găsesc modelul la '{MODEL_PATH}'. Verifică dacă ai rulat antrenarea!")
    st.stop()

input_image = None

# --- INPUT ---
if app_mode == "📸 Camera Live":
    camera_file = st.camera_input("Fă o poză deșeului")
    if camera_file is not None:
        input_image = Image.open(camera_file)

elif app_mode == "📂 Încărcare Fișier":
    uploaded_file = st.file_uploader("Upload imagine test...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)

# --- PROCESARE ---
if input_image is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Imagine Intrare")
        st.image(input_image, use_container_width=True)

    with col2:
        st.subheader("Rezultat Inferență")
        
        # 1. Convertim la RGB (rezolvă problema imaginilor PNG cu transparență)
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        # 2. Resize la 224x224 (Standard MobileNet)
        img_resized = input_image.resize((224, 224))
        
        # 3. Conversie la array
        img_array = np.array(img_resized)
        
        # 4. Adăugăm dimensiunea batch-ului (devine 1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 5. PREPROCESARE MOBILENET (CRUCIAL!)
        # Asta transformă pixelii exact cum a învățat modelul (-1 la 1)
        img_array = preprocess_input(img_array)

        # 6. Predicție
        predictions = model.predict(img_array)
        score = predictions[0]
        max_conf = np.max(score)
        label_idx = np.argmax(score)
        label_name = CLASSES[label_idx]

        # 7. Afișare
        bin_info = get_bin_info(label_name)
        result_str = f"{bin_info['icon']} Detectat: **{label_name.upper()}**"

        if bin_info['color'] == 'blue':
            st.info(result_str)
            st.info(f"🚮 {bin_info['bin_name']}")
        elif bin_info['color'] == 'green':
            st.success(result_str)
            st.success(f"🚮 {bin_info['bin_name']}")
        elif bin_info['color'] == 'gold':
            st.warning(result_str)
            st.warning(f"🚮 {bin_info['bin_name']}")
        else:
            st.error(result_str)
            st.error(f"🚮 {bin_info['bin_name']}")

        st.caption(f"💡 Instrucțiune: {bin_info['msg']}")

        st.markdown("---")
        st.write("📊 **Încredere Model (Confidence):**")
        
        # Bară de progres colorată
        my_bar = st.progress(0)
        my_bar.progress(int(max_conf * 100))
        st.write(f"Confidence: **{max_conf*100:.2f}%**")

        # Detalii tehnice pentru profesor
        with st.expander("🛠️ Vezi date tehnice (Debug Info)"):
            st.json({
                "Model": "MobileNetV2 (Transfer Learning)",
                "Input Shape": "(1, 224, 224, 3)",
                "Probabilități": {
                    c: f"{s*100:.2f}%" for c, s in zip(CLASSES, score)
                }
            })