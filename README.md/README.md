# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date

**Disciplina:** Rețele Neuronale
**Proiect:** Sistem Inteligent pentru Clasificarea Automată a Deșeurilor (Garbage Classification)
**Student:** [Paun Ionut-Andrei]
**Grupa:** [634 AB]
**Data** [20.11.2025]

---

## 1. Introducere
Acest document detaliază activitățile realizate în **Etapa 3**, concentrată pe achiziția datelor, analiza exploratorie (EDA) și preprocesarea necesară pentru antrenarea rețelelor neuronale. S-a utilizat un script Python automatizat pentru separarea datelor și generarea statisticilor.

---

## 2. Structura Repository-ului

```text
project-garbage-ai/
├── README.md                  # Documentația curentă
├── requirements.txt           # Lista dependențelor (matplotlib)
├── docs/
│   └── images/
│       └── distributie_clase.png  # Graficul generat automat
├── data/
│   ├── raw/                   # Datele originale descărcate
│   ├── train/                 # 70% date antrenare (generat)
│   ├── validation/            # 15% date validare (generat)
│   └── test/                  # 15% date testare (generat)
└── src/
    └── prepare_data.py        # Scriptul Python pentru split și EDA