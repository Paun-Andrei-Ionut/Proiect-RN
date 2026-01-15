# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Paun Ionut-Andrei]  
**Link Repository GitHub:** [LINK-UL TAU AICI]  
**Data predării:** 

---

## Scopul Etapei 6

Această etapă finalizează ciclul de dezvoltare al Sistemului cu Inteligență Artificială (SIA) pentru sortarea deșeurilor. Obiectivele principale au fost optimizarea modelului pentru a atinge o acuratețe de peste 70%, rezolvarea problemelor de confuzie între clasele similare (Metal vs. Sticlă) și maturizarea aplicației software prin introducerea unui sistem vizual de ghidare a utilizatorului (coșuri colorate).

---

## 1. Actualizarea Aplicației Software în Etapa 6

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6 (Final)** | **Justificare** |
|----------------|-------------------|--------------------------------|-----------------|
| **Interfață Utilizator** | Afișare text simplu | **Coduri de culoare (Albastru/Galben/Verde/Roșu)** | Ghidare vizuală rapidă pentru utilizator conform standardelor de reciclare. |
| **Metoda de Input** | Doar Upload Fișier | **Upload + Camera Live** | Permite utilizarea aplicației pe mobil/tabletă în timp real. |
| **Afișare Rezultat** | Clasa prezisă | **Clasă + Confidence Bar** | Utilizatorul vede cât de "sigur" este modelul pe decizia luată. |
| **Mesaje Educative** | Lipsă | **Instrucțiuni specifice** | Ex: "Scoate dopul", "Spală recipientul" – rol educativ. |
| **Model Backend** | `trained_model.h5` | `model_final.keras` (Optimizat) | Versiunea optimizată are o stabilitate mai mare pe clasele Paper și Plastic. |

### Diagrama Fluxului de Date (State Machine)

Fluxul final al aplicației este:
1.  **Input:** Imagine (Webcam sau Upload)
2.  **Preprocesare:** Resize (224, 224) -> Normalizare (0-1)
3.  **Inferență:** Model CNN -> Vector de probabilități
4.  **Logică Decizie:** Alegere clasa cu `max(score)`
5.  **Output:** Afișare Coș (Culoare) + Mesaj Educativ + Bară Siguranță

---

## 2. Analiza Detaliată a Performanței

### 2.1 Confusion Matrix și Interpretare

**Fișier:** `docs/images/confusion_matrix.png`

**Analiza rezultatelor pe setul de test (Test Set):**
1.  **Acuratețe Generală:** **70.23%** (Target atins).
2.  **Clasa "Paper" (Hârtie):** Este recunoscută cel mai bine (aprox. 85% precizie). Textura mată și formele plate o fac ușor de distins.
3.  **Problema Metal vs. Sticlă:** * Matricea arată o confuzie semnificativă între **Metal** și **Sticlă**.
    * *Cauza:* Reflexiile luminoase (specular highlights) apar pe ambele materiale. La rezoluția de 224x224, rețeaua confundă luciul dozelor de aluminiu cu transparența sticlei.
4.  **Clasa "Plastic":** Performanță medie, fiind uneori confundată cu sticla (în cazul PET-urilor transparente).

### 2.2 Analiza a 5 Exemple Greșite (Error Analysis)

| Index | Imagine Reală | Predicție | Confidence | Cauză Probabilă | Soluție Viitoare |
|---|---|---|---|---|---|
| **#1** | **Metal** (Doză) | Glass | 58% | Reflexie puternică a blițului pe metal, interpretată ca luciu de sticlă. | Augmentare cu variații de luminozitate (Brightness). |
| **#2** | **Paper** (Revistă) | Metal | 45% | Hârtie lucioasă care reflectă lumina, pierzând textura de "mat". | Antrenare pe imagini cu hârtie lucioasă. |
| **#3** | **Glass** (Sticlă) | Plastic | 62% | Sticlă mată/murdară, opacă, semănând cu un bidon de plastic. | Colectare date diversificate (sticlă murdară). |
| **#4** | **Plastic** (PET) | Glass | 51% | Plastic transparent perfect, vizual identic cu sticla în poză statică. | Creșterea rezoluției input (ex: 512x512). |
| **#5** | **Metal** (Spray) | Plastic | 49% | Eticheta de plastic acoperea tot metalul tubului. | Etichetare mai atentă a obiectelor mixte. |

---

## 3. Optimizarea Parametrilor și Experimentare

### Tabel Experimente de Optimizare

| **Exp#** | **Configurație** | **Accuracy** | **F1-score** | **Observații** |
|----------|------------------|--------------|--------------|----------------|
| **Baseline** | CNN Simplu (2 straturi Conv), 10 Epoci | 0.55 | 0.51 | Underfitting, modelul nu învăța trăsăturile complexe. |
| **Exp 1** | Adăugare Dropout (0.5) | 0.62 | 0.58 | A redus overfitting-ul, dar acuratețea a rămas scăzută. |
| **Exp 2** | Curățare Date (Eliminare clase Trash/Cardboard) | 0.68 | 0.65 | Eliminarea claselor cu puține date a stabilizat modelul. |
| **Final** | **CNN + Augmentare + 4 Clase Clare** | **0.70** | **0.69** | **Model Ales.** Cel mai bun echilibru între precizie și viteză. |

**Configurația Finală:** Am ales modelul din ultimul experiment deoarece oferă cea mai mare robustețe pentru aplicația reală, eliminând clasele care introduceau "zgomot" (Trash) și concentrându-se pe cele 4 materiale reciclabile principale.

---

## 4. Concluzii Finale și Lecții Învățate

### 4.1 Evaluare Sintetică
Proiectul a atins obiectivele propuse, rezultând într-un asistent de reciclare funcțional. Aplicația web integrează cu succes modelul de Deep Learning, oferind feedback în timp real prin camera video sau upload.

### 4.2 Limitări Identificate
* **Dependența de fundal:** Modelul performează mai slab dacă obiectul este ținut într-o mână sau are un fundal foarte colorat, deoarece a fost antrenat preponderent pe fundaluri neutre.
* **Confuzii Materiale:** Distincția dintre un PET transparent (Plastic) și o sticlă transparentă (Glass) rămâne dificilă doar pe baza informației vizuale RGB.

### 4.3 Lecții Învățate
1.  **Datele > Modelul:** Am petrecut mult timp optimizând codul rețelei, dar cel mai mare salt de performanță (de la 62% la 70%) a venit când am curățat datele și am eliminat clasele redundante.
2.  **Structura Proiectului:** Erorile de tip "File Not Found" m-au învățat importanța gestionării corecte a căilor (`os.path`) și a structurii folderelor (`src`, `data`, `models`).
3.  **UX contează:** Un model bun este inutil dacă utilizatorul nu înțelege rezultatul. Adăugarea culorilor specifice coșurilor a transformat un proiect tehnic într-un produs utilizabil.

---

### Instrucțiuni de Rulare a Proiectului Final

1.  **Instalare dependențe:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Rulare Aplicație Web:**
    ```bash
    streamlit run src/app/web_app.py
    ```
    *Aplicația se va deschide automat în browser.*