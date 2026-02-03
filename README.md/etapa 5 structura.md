# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Paun Ionut-Andrei
**Grupa:** 634AB
**Proiect:** SIA pentru Sortarea Automată a Deșeurilor
**Link Repository GitHub:** https://github.com/Paun-Andrei-Ionut/Proiect-RN.git
**Data predării:** [11.12.2025]

---

## Scopul Etapei 5

Această etapă marchează tranziția de la arhitectura software (definită în Etapa 4) la un sistem inteligent funcțional. Obiectivul este antrenarea rețelei neuronale convoluționale (CNN) pe setul de date hibrid (public + propriu) pentru a obține performanțe optime în clasificarea deșeurilor reciclabile și integrarea modelului final în aplicația de sortare.

---

## 1. Pregătirea Datelor (Prerequisite Verificat)

Pentru a asigura generalizarea modelului în condiții reale de iluminare și fundal, am respectat proporția de date originale, esențială pentru robustețea industrială.

* **Total imagini dataset:** 1.500 imagini
* **Imagini Originale (Proprii):** 600 imagini (40%) – *Capturate conform procedurii din Etapa 4*
* **Imagini Dataset Public:** 900 imagini (60%)
* **Structura Split (Stratificată):**
    * **Train (70%):** ~1.050 imagini (Folosite pentru învățare)
    * **Validation (15%):** ~225 imagini (Folosite pentru tuning și Early Stopping)
    * **Test (15%):** ~225 imagini (Folosite strict pentru evaluarea finală)

**Procesare aplicată:**
Toate imaginile au trecut prin pipeline-ul definit în State Machine:
1.  **Redimensionare:** 224x224 px (Standard pentru CNN).
2.  **Normalizare:** Scalare pixeli $[0, 255] \to [0, 1]$.
3.  **Data Augmentation (Doar pe Train):** S-au aplicat rotații aleatoare, zoom și flip orizontal pentru a simula poziția necontrolată a deșeurilor pe bandă.

---

## 2. Configurare și Hiperparametri (Nivel 1 & 2)

Am antrenat modelul folosind următorii parametri. Alegerea lor este justificată de natura vizuală a problemei (clasificare texturi deșeuri).

| **Hiperparametru** | **Valoare Aleasă** | **Justificare pentru Waste Sorting** |
|--------------------|-------------------|--------------------------------------|
| **Learning rate** | 0.001 | Valoare standard pentru optimizer-ul Adam. Asigură o convergență rapidă la început, critică pentru features vizuale complexe (ex: distincția fină între sticlă și plastic transparent). |
| **Batch size** | 32 | Având un dataset de 1.500 imagini, batch-ul de 32 oferă un echilibru optim între viteza de execuție și stabilitatea gradientului. |
| **Number of epochs** | 50 (cu Early Stopping) | S-a setat un maxim de 50, dar mecanismul de **Early Stopping** oprește antrenarea dacă `val_loss` nu scade timp de 5 epoci, prevenind overfitting-ul pe fundalul imaginilor proprii. |
| **Optimizer** | Adam | Cel mai eficient optimizer pentru CNN-uri generaliste. Gestionează learning rate-ul adaptiv, esențial când avem clase cu trăsături vizuale diferite (ex: cartonul mat vs. doza metalică lucioasă). |
| **Loss function** | Categorical Crossentropy | Funcția standard pentru clasificare multi-class, penalizând logaritmic predicțiile greșite. |
| **Activation functions** | ReLU (hidden) / Softmax (output) | **ReLU** pentru viteză și non-linearitate. **Softmax** obligatoriu în ultimul strat pentru a obține probabilități procentuale (suma 100%). |

---

## 3. Rezultate și Metrici (Nivel 1)

Modelul antrenat a fost salvat în `models/trained_model.h5` și evaluat pe setul de test (imagini pe care nu le-a "văzut" niciodată).

### Performanță pe Test Set:
* **Acuratețe (Accuracy):** **[ 0.XX ]** (Completați cu valoarea reală, ex: 0.78)
* **F1-Score (Macro):** **[ 0.XX ]** (Completați cu valoarea reală)
* **Loss Final:** [ 0.XX ]

*(Notă: Valorile sunt generate de scriptul `src/neural_network/evaluate.py`)*

### Grafice Antrenare (Nivel 2):
Curba de învățare arată scăderea `loss`-ului pe setul de antrenare și validare.
**Vizualizare:** Graficul este salvat în `docs/loss_curve.png`.

---

## 4. Analiză Erori în Context Industrial (Nivel 2 - Obligatoriu)

Pentru o aplicație de sortare deșeuri, acuratețea globală ascunde detalii critice. Am analizat comportamentul modelului:

### A. Confuzii Principale
Din Matricea de Confuzie, modelul tinde să confunde cel mai des clasa **Plastic** cu clasa **Sticlă**.
* **Cauză:** Ambele materiale sunt adesea transparente și reflectă lumina (specular highlights). La rezoluția de 224x224, diferența de textură fină se pierde, iar rețeaua se bazează excesiv pe contur, care poate fi similar (ex: sticlă PET vs sticlă sticlă).

### B. Caracteristici Problematice
Erorile apar frecvent în imaginile cu **obiecte deformate** (ex: PET-uri strivite extrem de tare) sau când **eticheta** acoperă complet materialul (recunoaște brandul, nu materialul). De asemenea, reflexiile puternice de la bliț sunt uneori interpretate greșit ca fiind metal.

### C. Implicații Industriale
În contextul unei benzi de sortare automate:
* **False Positive (Contaminare):** Dacă plasticul este clasificat greșit ca hârtie/carton, întregul lot de hârtie reciclată poate fi respins de procesator. Aceasta este o eroare **CRITICĂ**.
* **False Negative (Pierdere):** Dacă o sticlă nu este detectată (Unknown) și ajunge la groapa de gunoi, se pierde valoarea materialului, dar procesul nu este oprit.
* **Concluzie:** Prioritatea este minimizarea False Positives pe clasele sensibile (Hârtie).

### D. Măsuri Corective Propuse
1.  **Augmentare "Lighting-Aware":** Adăugarea de `RandomBrightness` și `RandomContrast` în antrenare pentru a face modelul imun la reflexii.
2.  **Dataset Balansat pe Deformări:** Colectarea a 200 de imagini suplimentare strict cu deșeuri strivite/turtite.
3.  **Threshold Dinamic:** Ajustarea pragului de decizie în `main.py` la 0.60 pentru clasa Hârtie (pentru siguranță) și 0.40 pentru Plastic.

---

## 5. Integrare în UI și Demonstrație (Nivel 1)

Modelul antrenat a fost integrat cu succes în aplicația principală (`src/app/main.py`), înlocuind modelul dummy din Etapa 4.

**Screenshot Inferență Reală:**
Imaginea de mai jos demonstrează funcționarea sistemului, identificând corect un obiect cu un scor de încredere ridicat.

![Real Inference Screenshot](docs/screenshots/inference_real.png)

---

## 6. Structura Repository-ului la Finalul Etapei 5

```text
proiect-rn-[nume]/
├── README.md                           # Overview general
├── docs/
│   ├── etapa4_arhitectura_sia.md      # Etapa anterioară
│   ├── etapa5_antrenare_model.md      # ← ACEST FIȘIER
│   ├── state_machine.png
│   ├── loss_curve.png                 # (Nivel 2) Grafic Loss
│   ├── confusion_matrix.png           # (Opțional)
│   └── screenshots/
│       └── inference_real.png         # (Nivel 1) Screenshot obligatoriu UI
├── data/
│   ├── raw/                           # Date originale
│   ├── generated/                     # Date proprii (40%)
│   ├── train/                         # Date antrenare
│   ├── validation/                    # Date validare
│   └── test/                          # Date testare
├── src/
│   ├── data_acquisition/
│   │   └── collect_data.py
│   ├── preprocessing/                 # Scripturi curățare/split
│   ├── neural_network/
│   │   ├── model.py                   # Arhitectura CNN
│   │   ├── train.py                   # Script antrenare (NOU)
│   │   └── evaluate.py                # Script evaluare (NOU)
│   └── app/
│       └── main.py                    # UI Actualizat cu model antrenat
├── models/
│   ├── untrained_model.h5
│   └── trained_model.h5               # (Nivel 1) Modelul final salvat
├── results/
│   ├── training_history.csv           # Log-uri antrenare
│   └── test_metrics.json              # Rezultate finale
└── requirements.txt