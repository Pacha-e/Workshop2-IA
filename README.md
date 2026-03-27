#  Workshop 2 — Machine Learning & Deep Learning Aplicado

Sistema de resolución de problemas supervisados de clasificación y regresión, implementado con scikit-learn y PyTorch.

## 📋 Tabla de contenidos
- [Descripción general](#descripción-general)
- [Tecnologías utilizadas](#tecnologías-utilizadas)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Problema 1: Clasificación](#problema-1-clasificación)
- [Problema 2: Regresión](#problema-2-regresión)
- [Resultados](#resultados)
- [Instalación y ejecución](#instalación-y-ejecución)
- [Conclusiones](#conclusiones)
- [Autor](#autor)

---

## Descripción general

Este proyecto aborda dos problemas de Machine Learning supervisado:

1. Clasificación — detección de fatiga muscular usando señales EMG  
2. Regresión — estimación de edad a partir de imágenes faciales  

Cada problema cubre el pipeline completo:
EDA → preprocesamiento → entrenamiento → evaluación → análisis

---

## Tecnologías utilizadas

| Tecnología | Rol |
| :--- | :--- |
| Python 3.10 | Lenguaje base |
| PyTorch | Deep Learning |
| scikit-learn | Modelos clásicos |
| NumPy / Pandas | Manejo de datos |
| Matplotlib / Seaborn | Visualización |
| HuggingFace | Dataset EMG |
| Kaggle | Dataset imágenes |

---

## Estructura del proyecto

    workshop_2/
    ├── README.md
    ├── clasificacion/
    │   └── clasificacion.ipynb
    └── regresion/
        └── regresion.ipynb

---

## Problema 1: Clasificación

### Dataset

| Atributo | Valor |
| :--- | :--- |
| Nombre | Muscle Fatigue Cycling |
| Fuente | HuggingFace |
| Tipo | Señales EMG |
| Sujetos | 3 |
| Frecuencia | 1000 Hz |

### Metodología

- Mapeo de clases (3 → 2)  
- Ventanas de 1 segundo  
- 7 features por canal  
- EDA completo  
- División 70 / 15 / 15  
- Entrenamiento con múltiples modelos  

---

## Problema 2: Regresión

### Dataset

| Atributo | Valor |
| :--- | :--- |
| Nombre | UTKFace |
| Fuente | Kaggle |
| Tamaño | 3,250 imágenes |
| Edad | 1–116 |

### Metodología

- EDA de edades  
- Visualización de imágenes  
- Data augmentation  
- División 70 / 15 / 15  
- CNN para regresión  
- Mejora con ResNet18  

---

## Resultados

### Clasificación

| Modelo | Accuracy | F1 |
| :--- | :--- | :--- |
| kNN | 0.8647 | 0.7510 |
| Decision Tree | 0.8315 | 0.6960 |
| Random Forest | 0.9091 | 0.8340 |
| Gradient Boosting | 0.9069 | 0.8346 |
| DNN | 0.8869 | 0.8090 |

### Regresión
| Modelo | MAE | RMSE | R² | Nota |
| :--- | :--- | :--- | :--- | :--- |
| CNN v1 | 10.21 | 13.97 | 0.369 | Requerido por el enunciado |
| CNN v2 | 9.77 | 13.41 | 0.419 | Mejora adicional |
| ResNet18 | 7.58 | 10.19 | 0.664 | Mejora adicional |
---

## Instalación y ejecución

### Requisitos
- Python 3.10  
- Conda  

### Instalación

    conda create -n workshop2 python=3.10 -y
    conda activate workshop2

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install scikit-learn scipy matplotlib seaborn pandas numpy jupyter kagglehub datasets huggingface_hub tqdm

### Ejecución

    conda activate workshop2

Problema 1:

    cd clasificacion
    jupyter notebook clasificacion.ipynb

Problema 2:

    cd ../regresion
    jupyter notebook regresion.ipynb

---

## Conclusiones

Problema 1:
- Random Forest fue el mejor modelo (F1 = 0.834)  
- La fatiga muscular se detecta con features espectrales  
- Limitación: dataset pequeño  

Problema 2:
- CNN baseline cumple el objetivo (MAE = 10.21)  
- Transfer Learning mejora significativamente:  
  MAE: 10.21 → 7.58  
  R²: 0.369 → 0.664  

---

## Autor

Emmanuel Hernández
Universidad EAFIT — Introducción a la Inteligencia Artificial 2026-01
