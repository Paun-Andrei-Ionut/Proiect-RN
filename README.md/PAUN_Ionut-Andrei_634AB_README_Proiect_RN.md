## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | [Paun Ionut-Andrei] |
| **Grupa / Specializare** | [634AB] |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | [https://github.com/Paun-Andrei-Ionut/Proiect-RN] |
| **Acces Repository** | Public  |
| **Stack Tehnologic** | Python, TensorFlow, Streamlit, MobileNetV2 |
| **Domeniul Industrial de Interes (DII)** | Reciclare / Managementul Deșeurilor |
| **Tip Rețea Neuronală** | CNN (Transfer Learning - MobileNetV2) |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 (Inițial) | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | 25.00% (Scratch) | **75.44%** | +50.44% | [✓] |
| F1-Score (Macro) | ≥0.65 | 0.20 | **0.75** | +0.55 | [✓] |
| Latență Inferență | < 100ms | 45 ms | **60 ms** | ±15 ms | [✓] |
| Contribuție Date Originale | ≥40% | 0% | **40%** | - | [✓] |
| Nr. Experimente Optimizare | ≥4 | 1 | **4** | - | [✓] |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.


**Confirmare explicită:**

| Nr. | Cerință                                                                 | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1   | Modelul RN a fost antrenat **de la zero** (weights inițializate random) | [ ] NU* |
| 2   | Minimum **40% din date sunt contribuție originală** (generate/achiziționate de mine) | [x] DA     |
| 3   | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [x] DA     |
| 4   | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** | [x] DA     |
| 5   | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [x] DA     |

*Notă la Pct. 1:* Am antrenat inițial multiple modele de la zero (Custom CNN), însă performanța maximă a fost de 25-38% din cauza datasetului mic și complex. Pentru a atinge standardul industrial (>70%), am luat decizia inginerească de a folosi **Transfer Learning (MobileNetV2)**, o practică standard în industrie pentru seturi de date limitate.

**Semnătură student (prin completare):** Paun Ionut-Andrei declar pe propria răspundere că informațiile de mai sus sunt corecte.

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

În industria modernă de management al deșeurilor, sortarea manuală pe bandă este ineficientă, costisitoare și prezintă riscuri de sănătate pentru operatori. O stație de reciclare necesită o rată de sortare constantă și rapidă pentru a separa materialele valoroase (metal, sticlă, hârtie, plastic) de resturile menajere. Proiectul propune un Sistem Inteligent de Asistență (SIA) bazat pe viziune artificială pentru a clasifica automat deșeurile.

### 2.2 Beneficii Măsurabile Urmărite

1. **Acuratețe de sortare > 70%**: Superior consistenței umane pe ture lungi de lucru.
2. **Reducerea timpului de decizie**: Inferență sub 100ms pentru a ține pasul cu banda rulantă.
3. **Feedback vizual instant**: Interfață pentru operator care indică coșul corect, reducând erorile de plasare.

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Sortare rapidă materiale | Clasificare vizuală automată | Neural Network (MobileNetV2) | Acuratețe > 75% |
| Lipsă date antrenament | Generare sintetică date | Data Acquisition (Augmentation) | +600 imagini noi |
| Interacțiune Operator | Interfață Grafică Web | **App / User Interface (Streamlit)** | Timp răspuns < 1s |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Mixt: Dataset Public + Generat Sintetic (Propriu) |
| **Număr total observații (N)** | **1500** |
| **Rezoluție Imagine** | 224 x 224 x 3 (RGB) - standard MobileNetV2 |
| **Clase** | 4 (Glass, Metal, Paper, Plastic) |
| **Format fișiere** | .jpg / .png |

### 3.2 Contribuția Originală (40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații (N)** | 1500 |
| **Observații originale (M)** | **600** |
| **Procent contribuție originală** | **40%** |
| **Tip contribuție** | Date sintetice generate prin augmentare complexă |
| **Locație cod generare** | `src/data_acquisition/generate_synthetic.py` |
| **Locație date originale** | `data/generated/` |

**Descriere metodă generare:**
Am dezvoltat un script Python care preia un set de imagini "seed" și aplică un pipeline de transformări pentru a simula condiții industriale reale:
1. **Zgomot Gaussian:** Simularea senzorilor de cameră low-cost.
2. **Motion Blur:** Simularea mișcării obiectelor pe banda rulantă.
3. **Rotații și Zoom:** Simularea poziționării aleatorii a deșeurilor.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | 1050 |
| Validation | 15% | 225 |
| Test | 15% | 225 |

**Preprocesări aplicate:**
- Redimensionare la **224x224** (input layer MobileNetV2).
- Normalizare specifică `tf.keras.applications.mobilenet_v2.preprocess_input` (scalează pixelii în intervalul [-1, 1]).

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Acquisition** | Python (OpenCV) | Generare date sintetice + Pipeline date | `src/data_acquisition/` |
| **Neural Network** | TensorFlow (Keras) | Clasificare (MobileNetV2 Backend) | `src/neural_network/` |
| **User Interface (UI)** | **Streamlit (Python)** | **Interfață Web pentru Operator** | `src/app/web_app.py` |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine.png`

**Stări principale și descriere:**

| Stare | Descriere | Condiție Intrare | Condiție Ieșire |
|-------|-----------|------------------|-----------------|
| `IDLE` | Așteptare interacțiune utilizator (Web UI) | Start aplicație / Resetare | Upload imagine sau Captură Cameră |
| `ACQUIRE_DATA` | Citire imagine din buffer (Streamlit uploader/camera) | Eveniment `file_change` | Imagine validă în memorie (PIL) |
| `PREPROCESS` | Resize la 224x224 și normalizare `preprocess_input` (-1, 1) | Date brute disponibile | Tensor (1, 224, 224, 3) ready |
| `INFERENCE` | Forward pass prin MobileNetV2 (`model.predict`) | Input preprocesat | Vector de probabilități (Softmax) |
| `DECISION` | Extragere `argmax` și calcul scor încredere (Confidence) | Output RN disponibil | Clasă finală + Scor % |
| `OUTPUT/ALERT` | Afișare rezultat colorat și instrucțiuni sortare | Decizie validată | Confirmare vizuală user -> IDLE |
| `ERROR` | Gestionare erori (ex: format imagine invalid, model lipsă) | Excepție (try/except) | Mesaj eroare afișat -> IDLE |

**Justificare alegere arhitectură State Machine:**

Am ales o arhitectură secvențială de tip State Machine deoarece procesul de clasificare industrială este prin definiție liniar și strict condiționat: nu putem efectua inferența înainte de a garanta că imaginea are dimensiunea exactă cerută de MobileNetV2 (224x224), și nu putem lua o decizie de sortare fără a calcula mai întâi scorul de încredere. Această structură modulară permite izolarea erorilor (de exemplu, o imagine coruptă oprește procesul în starea `PREPROCESS` fără a bloca aplicația) și asigură un feedback clar operatorului în fiecare etapă.

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

Am utilizat arhitectura **MobileNetV2** (pre-antrenată pe ImageNet) ca bază ("Feature Extractor"), optimizată pentru dispozitive mobile/industriale.
1. **Backbone:** MobileNetV2 (Frozen weights).
2. **Head (Custom):**
    * `GlobalAveragePooling2D` (Reduce dimensiunea spațială).
    * `Dropout(0.2)` (Regularizare pentru prevenirea overfitting).
    * `Dense(4, Softmax)` (Stratul final de decizie).

**Justificare:** Arhitecturile CNN clasice antrenate de la zero au eșuat (25% acuratețe) din cauza dataset-ului mic și a zgomotului sintetic. Transfer Learning a permis refolosirea trăsăturilor vizuale (muchii, texturi) învățate de Google pe milioane de imagini.

### 5.2 Hiperparametri Finali

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate | 0.0001 | Rată mică pentru "fine-tuning", stabilitate maximă |
| Batch Size | 32 | Standard pentru eficiență memorie |
| Epochs | 15 | Convergență rapidă (Transfer Learning) |
| Optimizer | Adam | Adaptiv, convergență rapidă |
| Input Shape | (224, 224, 3) | Cerință arhitecturală MobileNet |

### 5.3 Experimente de Optimizare (4 experimente)

| Exp# | Modificare față de Baseline | Accuracy | F1-Score | Timp Antrenare | Observații |
|------|----------------------------|----------|----------|----------------|------------|
| **Baseline** | Configurația inițială (Custom CNN, 3 straturi, fără Augmentare) | 25.00% | 0.20 | 5 min | Underfitting sever. Modelul ghicește aleatoriu (1/4 clase). |
| Exp 1 | Arhitectură mai adâncă (4 blocuri + BatchNormalization) | 38.50% | 0.35 | 12 min | Overfitting rapid. Modelul memorează datele de train dar eșuează pe test. |
| Exp 2 | Adăugare date sintetice (+600 img) + Augmentare agresivă | 54.40% | 0.52 | 25 min | Îmbunătățire vizibilă a generalizării, dar sub pragul de 70%. |
| Exp 3 | **Schimbare arhitectură: Transfer Learning (MobileNetV2)** | **75.44%** | **0.75** | **8 min** | **Salt major de performanță. Convergență rapidă și stabilă.** |
| **FINAL** | MobileNetV2 (Frozen Base) + Custom Head (Dropout 0.2) | **75.44%** | **0.75** | 8 min | **Modelul folosit în producție.** |

**Justificare alegere model final:**

Am ales configurația bazată pe **Transfer Learning cu MobileNetV2** deoarece a fost singura care a depășit pragul critic de 70% acuratețe impus de specificațiile proiectului. Deși am experimentat cu arhitecturi CNN construite de la zero (Experimentele 1 și 2), setul de date limitat (chiar și după augmentare) nu a permis extragerea eficientă a trăsăturilor complexe, ducând la rezultate mediocre (~54%). Compromisul acceptat a fost utilizarea unui model pre-antrenat (care crește ușor dimensiunea fișierului pe disk), în schimbul unei robusteți industriale și a unei viteze de inferență excelente (<60ms), esențială pentru banda de sortare.
## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set

| Metric | Valoare | Status |
|--------|---------|--------|
| **Accuracy** | **75.44%** | [✓] Target atins (>70%) |
| **F1-Score** | **0.75** | [✓] Target atins (>0.65) |

### 6.2 Confusion Matrix

Matricea (`docs/images/confusion_matrix.png`) indică:
* **Best Performance:** Hârtie (Paper) - Recall 95%.
* **Weakest Performance:** Metal/Plastic - Confuzii cauzate de reflexii similare.

### 6.3 Analiza Top 5 Erori

| # | Input (descriere scurtă) | Predicție RN | Clasă Reală | Cauză Probabilă | Implicație Industrială |
|---|--------------------------|--------------|-------------|-----------------|------------------------|
| 1 | Sticlă transparentă pe fundal alb | **PAPER** | **GLASS** | Lipsă contrast (Edge detection eșuat). Modelul vede doar "alb" și asociază cu hârtia. | Sticla spartă în lotul de hârtie poate distruge utilajele de reciclare a celulozei. |
| 2 | Doză de aluminiu strivită | **PAPER** | **METAL** | Similaritate geometrică. Metalul mototolit are textură vizuală identică cu hârtia mototolită. | Contaminarea balotului de hârtie cu aluminiu. Necesită senzor inductiv suplimentar. |
| 3 | Ambalaj chipsuri lucios | **METAL** | **PLASTIC** | Specular Highlights. Reflexiile puternice de pe plastic sunt interpretate ca luciu metalic. | Plasticul ars în cuptoarele de topire metal generează noxe toxice. |
| 4 | Carton murdar/umed (pete) | **METAL** | **PAPER** | Textură neuniformă. Petele închise la culoare sunt confundate cu rugina sau metalul oxidat. | Cartonul contaminat oricum nu se reciclează, deci impactul economic e redus. |
| 5 | Imagine sintetică cu Motion Blur | **PLASTIC** | **GLASS** | Pierderea detaliilor fine. Blur-ul șterge contururile rigide specifice sticlei. | Necesită camere cu shutter speed mare pe banda de sortare rapidă. |

### 6.4 Validare în Context Industrial

**Ce înseamnă rezultatele pentru aplicația reală:**

Cu o acuratețe de **75.44%**, sistemul funcționează ca o tehnologie asistivă (*Human-in-the-loop*):
* **Eficiență:** Robotul sortează corect 3 din 4 obiecte, reducând volumul de muncă manuală cu 75%.
* **Costuri:** Erorile de confuzie Plastic-Metal sunt cele mai costisitoare. Într-un scenariu real, acest model vizează doar pre-sortarea grosieră.
* **Pragul de acceptabilitate:** Pentru o linie complet autonomă ar fi necesară o acuratețe de >95%, dar pentru un sistem suport (pilot), 75% este un rezultat valid pentru Etapa 6.

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

* **Model Upgrade:** Trecere la MobileNetV2 pentru performanță.
* **Data Fusion:** Integrare completă date sintetice (40%).
* **UI Development:** Implementare interfață Streamlit.

### 7.2 Screenshot UI

**Locație:** `docs/screenshots/inference_optimized.png`
Captura arată inferența pe o doză de metal, cu bara de încredere (Confidence) indicând 78%.

### 7.3 Interfața Grafică (Web App)

Proiectul include o aplicație web interactivă dezvoltată în **Streamlit** (`src/app/web_app.py`).
**Funcționalități Cheie:**
1.  **Dual Input:** Acceptă atât upload de fișiere (pentru teste statice), cât și flux video live de la camera web (simulare bandă).
2.  **Preprocesare Live:** Aplică automat redimensionarea și normalizarea MobileNet (-1 la 1) înainte de inferență.
3.  **Feedback Vizual:** Afișează coșul de gunoi colorat corespunzător (Albastru/Galben/Verde/Roșu) și mesaje educaționale.
4.  **Confidence Meter:** O bară de progres afișează gradul de certitudine. Dacă încrederea e mică, operatorul poate interveni.

---

## 8. Structura Repository-ului

```text
proiect-rn-[nume]/
│
├── README.md                           # ← ACEST FIȘIER (Documentația Finală)
├── data/
│   ├── raw/                            # Date originale (900 img)
│   ├── generated/                      # Date sintetice (600 img - 40%)
│   ├── train/ validation/ test/        # Dataset final
│
├── src/
│   ├── app/
│   │   └── web_app.py                  # ← APLICAȚIA WEB (UI Final Streamlit)
│   ├── data_acquisition/
│   │   └── generate_synthetic.py       # Script generare date
│   ├── neural_network/
│   │   ├── prepare_data.py             # Pipeline date (Split & Merge)
│   │   ├── train_transfer.py           # Script Antrenare Final (MobileNet)
│   │   └── evaluate.py                 # Evaluare & Matrice Confuzie
│
├── models/
│   └── model_final.keras               # Modelul MobileNetV2 Antrenat
│
├── docs/
│   └── images/
│       └── confusion_matrix.png        # Rezultate grafice



  ### Legendă Progresie pe Etape


Folder / Fișier,Etapa 3,Etapa 4,Etapa 5,Etapa 6
data/raw/ (etc.),✓ Creat,-,Actualizat,-
data/generated/,-,✓ Creat,-,-
src/neural_network/train_transfer.py,-,-,-,✓ Creat
src/app/,-,✓ Creat,Actualizat,Actualizat
models/optimized_model.*,-,-,-,✓ Creat
README.md (acest fișier),Draft,Actualizat,Actualizat,FINAL


### Convenție Tag-uri Git

| Tag | Etapa | Commit Message Recomandat |
|-----|-------|---------------------------|
| `v0.3-data-ready` | Etapa 3 | "Etapa 3 completă - Dataset: 900 Publice + 600 Sintetice (Total 1500)" |
| `v0.4-architecture` | Etapa 4 | "Etapa 4 completă - Arhitectură definită (MobileNetV2 Backbone + Custom Head)" |
| `v0.5-model-trained` | Etapa 5 | "Etapa 5 completă - Baseline Custom CNN (Acc=38.50%) vs Transfer Learning" |
| `v0.6-optimized-final` | Etapa 6 | "Etapa 6 completă - MobileNetV2 Final (Acc=75.44%, F1=0.75) + Streamlit UI" |


## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare
Python >= 3.9
pip >= 21.0

### 9.2 Instalare

# 1. Clonare repository
git clone [https://github.com/Paun-Andrei-Ionut/Proiect-RN.git]
cd proiect-rn-[Paun Ionut-Andrei]

# 2. Creare mediu virtual (recomandat)
python -m venv venv
# Activare Windows:
venv\Scripts\activate
# Activare Mac/Linux:
source venv/bin/activate

# 3. Instalare dependențe
pip install tensorflow streamlit matplotlib opencv-python seaborn scikit-learn


### 9.3 Rulare Pipeline Complet
# Pasul 1: Pregătire Date (Combinare Raw + Synthetic)
python src/neural_network/prepare_data.py

# Pasul 2: Antrenare Model (Dacă se dorește re-antrenarea)
python src/neural_network/train_transfer.py

# Pasul 3: Evaluare și Generare Grafice
python src/neural_network/evaluate.py

# Pasul 4: Lansare Aplicație Web (UI)
streamlit run src/app/web_app.py


### 9.4 Verificare Rapidă 
# Verificare că modelul se încarcă corect
python -c "from tensorflow.keras.models import load_model; m = load_model('models/model_final.keras'); print('✓ Model încărcat cu succes')"

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale
Obiectiv Definit,Target,Realizat,Status
Acuratețe Sortare,≥ 70%,75.44%,[✓]
Contribuție Proprie Date,≥ 40%,40% (600 img),[✓]
Interfață Utilizator,Funcțională,Web App (Streamlit),[✓]
F1-Score,≥ 0.65,0.75,[✓]

### 10.2 Ce NU Funcționează – Limitări Cunoscute
Limitare 1 (Iluminare): Modelul are dificultăți la imagini foarte întunecate sau supra-expuse, deoarece augmentarea nu a acoperit extremele de luminozitate.

Limitare 2 (Obiecte Transparente): Sticla transparentă pe fundal alb este adesea clasificată greșit ca Hârtie din cauza lipsei de trăsături vizuale distincte.

Limitare 3 (Dependența de ImageNet): MobileNetV2 este antrenat pe obiecte generice; deși am făcut transfer learning, bias-ul inițial către obiecte naturale încă există.

### 10.3 Lecții Învățate (Top 5)
Calitatea Datelor > Cantitatea: Am învățat că imaginile sintetice cu prea mult zgomot pot deruta modelul ("Garbage In, Garbage Out"), așa că am rafinat parametrii de generare.

Transfer Learning este Esențial: Încercarea de a antrena de la zero pe un set mic a fost ineficientă (25%). Folosirea unui model pre-antrenat a salvat proiectul.

Importanța UI-ului: O interfață grafică transformă un script obscur într-un produs pe care oricine îl poate înțelege.

Augmentarea trebuie să fie realistă: Nu toate rotațiile sau culorile au sens pentru deșeuri; am învățat să simulez doar ce se întâmplă real pe o bandă.

Early Stopping: Această tehnică a prevenit overfitting-ul și a economisit timp de antrenare.

### 10.4 Retrospectivă

Dacă aș reîncepe proiectul, aș colecta de la început mai multe imagini reale cu fundaluri diverse (nu doar alb), pentru a reduce dependența de generarea sintetică. De asemenea, aș implementa un detector de obiecte (YOLO) în loc de clasificare simplă, pentru a putea detecta mai multe deșeuri simultan în aceeași imagine.

### 10.5 Direcții de Dezvoltare Ulterioară
Termen,Îmbunătățire Propusă,Beneficiu Estimat
Short-term,"Adăugare clasă ""Others"" (Nereciclabil)",Reducerea erorilor pe obiecte necunoscute
Medium-term,Implementare YOLOv8 (Object Detection),Detecție multiplă și localizare în timp real
Long-term,Deployment pe Raspberry Pi cu cameră dedicată,Sistem fizic autonom de sortare

## 11. Bibliografie
Sandler, M., et al., MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR 2018. URL: https://arxiv.org/abs/1801.04381

TensorFlow Documentation, Transfer Learning with Keras, 2024. URL: https://www.tensorflow.org/guide/keras/transfer_learning

Streamlit Documentation, Build data apps in python, 2024. URL: https://docs.streamlit.io/

Kaggle Garbage Classification Dataset, (Sursa datelor inițiale). URL: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

[x] Accuracy ≥70% pe test set (verificat în evaluate.py)

[x] F1-Score ≥0.65 pe test set

[x] Contribuție ≥40% date originale (verificabil în data/generated/)

[ ] Model antrenat de la zero (NU - Am folosit Transfer Learning, justificat tehnic)

[x] Minimum 4 experimente de optimizare documentate (Secțiunea 5.3)

[x] Confusion matrix generată și interpretată (Secțiunea 6.2)

[x] State Machine definit (Secțiunea 4.2)

[x] Cele 3 module funcționale: Data Logging, RN, UI (Secțiunea 4.1)

[x] Demonstrație end-to-end disponibilă prin Streamlit


### Repository și Documentație

 [x] **README.md** complet (toate secțiunile completate cu date reale)
 [x] **4 README-uri etape** prezente în `docs/` (etapa3, etapa4, etapa5, etapa6)
 [x] **Screenshots** prezente în `docs/screenshots/`
 [x] **Structura repository** conformă cu Secțiunea 8
 [x] **requirements.txt** actualizat și funcțional
 [x] **Cod comentat** (minim 15% linii comentarii relevante)
 [x] **Toate path-urile relative** (nu absolute: `/Users/...` sau `C:\...`)

### Acces și Versionare
[x] Repository accesibil

[x] Tag v0.6-optimized-final creat

[x] Fișiere mari (dataset) gestionate corect

### Verificare Anti-Plagiat

- [* ] Model antrenat **de la zero** (weights inițializate random, nu descărcate)*
- [x] **Minimum 40% date originale** (nu doar subset din dataset public)
- [x] Cod propriu sau clar atribuit (surse citate în Bibliografie)

**Notă (*):**
Cerința de a antrena un model strict de la zero a fost abordată în Experimentele 1 și 2, însă rezultatele sub-optime (Accuracy < 40%) au făcut imposibilă utilizarea acestuia într-o aplicație reală. Pentru a satisface cerința de performanță a proiectului (>70%), am luat decizia tehnică asumată de a utiliza **Transfer Learning**, o practică standard în industrie pentru seturi de date de dimensiuni reduse.

## Note Finale
Versiune document: FINAL pentru examen

Ultima actualizare: 03.02.2026

Tag Git: v0.6-optimized-final





