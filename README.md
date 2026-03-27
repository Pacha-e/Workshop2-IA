\# Workshop 2 — Machine Learning \& Deep Learning Aplicado



\*\*Universidad EAFIT — Introducción a la Inteligencia Artificial (2026-01)\*\*



\---



\## Descripción General



Este workshop integra dos problemas supervisados independientes:



\- \*\*Problema 1 — Clasificación:\*\* Detección de fatiga muscular en ciclismo a partir de señales EMG.

\- \*\*Problema 2 — Regresión:\*\* Estimación de edad a partir de imágenes faciales usando Redes Neuronales Convolucionales.



Cada problema cubre el ciclo completo de un proyecto de Machine Learning: análisis del problema, exploración de datos, preprocesamiento, entrenamiento, evaluación y análisis crítico de resultados.



\---



\## Estructura del repositorio



&#x20;   workshop\_2/

&#x20;   ├── README.md

&#x20;   ├── clasificacion/

&#x20;   │   └── clasificacion.ipynb

&#x20;   └── regresion/

&#x20;       └── regresion.ipynb



\---



\## Problema 1 — Clasificación de Fatiga Muscular



\### Dataset

\- \*\*Nombre:\*\* Muscle Fatigue Cycling

\- \*\*Fuente:\*\* HuggingFace — YominE/Muscle\_Fatigue\_Cycling

\- \*\*Descripción:\*\* Señales EMG de 8 músculos de la pierna dominante de 3 sujetos realizando sprints en bicicleta a 1000 Hz.



\### Metodología

1\. Preprocesamiento del target — mapeo de 3 clases a 2 (normal vs desgaste)

2\. Extracción de características — ventanas de 1 segundo, 7 features por canal

3\. EDA completo — distribuciones, correlaciones, boxplots, balance de clases

4\. Pipeline con StandardScaler — división estratificada 70/15/15

5\. Entrenamiento y comparación de 5 modelos con Random Search



\### Resultados



| Modelo | Test Accuracy | Test F1 |

|---|---|---|

| kNN | 0.8647 | 0.7510 |

| Decision Tree | 0.8315 | 0.6960 |

| Random Forest | 0.9091 | 0.8340 |

| Gradient Boosting | 0.9069 | 0.8346 |

| DNN | 0.8869 | 0.8090 |



\*\*Mejor modelo:\*\* Random Forest — F1=0.834, Accuracy=0.909 en test. Reentrenado con train+val: F1=0.819, Accuracy=0.900.



\---



\## Problema 2 — Regresión de Edad con CNN



\### Dataset

\- \*\*Nombre:\*\* Faces: Age Detection from Images

\- \*\*Fuente:\*\* Kaggle — arashnic/faces-age-detection-dataset (UTKFace)

\- \*\*Descripción:\*\* 3,250 imágenes faciales etiquetadas con la edad del sujeto (1-116 años). Edad codificada en el nombre del archivo.



\### Metodología

1\. EDA — distribución de edades, visualización de muestras, análisis de sesgos

2\. Pipeline de preprocesamiento con data augmentation

3\. División aleatoria 70/15/15 — 2,275 / 487 / 488 imágenes

4\. Entrenamiento de CNN para regresión de edad



\### Resultado requerido por el enunciado



| Modelo | MAE | RMSE | R² |

|---|---|---|---|

| CNN v1 (64×64, desde cero) | 10.21 años | 13.97 años | 0.369 |



\### Mejoras adicionales (iniciativa propia)



Por fuera del alcance del enunciado, se exploraron dos mejoras progresivas para reducir el margen de error del modelo baseline:



| Modelo | MAE | RMSE | R² | Mejora sobre baseline |

|---|---|---|---|---|

| CNN v1 (64×64, baseline) | 10.21 años | 13.97 años | 0.369 | — |

| CNN v2 (96×96, mayor resolución) | 9.77 años | 13.41 años | 0.419 | -0.44 años |

| ResNet18 (128×128, Transfer Learning) | 7.58 años | 10.19 años | 0.664 | -2.63 años |



\*\*Principal hallazgo:\*\* Transfer Learning con ResNet18 preentrenado en ImageNet redujo el MAE en 2.63 años respecto al baseline y casi duplicó el R² — de 0.369 a 0.664. Esto demuestra que para datasets pequeños (3,250 imágenes) el transfer learning supera ampliamente a entrenar desde cero.



\---



\## Requisitos



&#x20;   Python 3.10

&#x20;   PyTorch 2.6 + CUDA 12.4

&#x20;   scikit-learn, scipy, matplotlib, seaborn, pandas, numpy

&#x20;   torchvision, kagglehub, datasets (HuggingFace)



\### Instalación



&#x20;   conda create -n workshop2 python=3.10 -y

&#x20;   conda activate workshop2

&#x20;   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

&#x20;   pip install scikit-learn scipy matplotlib seaborn pandas numpy jupyter kagglehub datasets huggingface\_hub tqdm



\---



\## Cómo ejecutar



&#x20;   conda activate workshop2



&#x20;   # Problema 1

&#x20;   cd clasificacion

&#x20;   jupyter notebook clasificacion.ipynb



&#x20;   # Problema 2

&#x20;   cd ../regresion

&#x20;   jupyter notebook regresion.ipynb



\---



\## Conclusiones



\*\*Problema 1:\*\* Random Forest fue el mejor clasificador con F1=0.834 sobre datos no vistos. La fatiga muscular es detectable con features espectrales — especialmente la frecuencia mediana que disminuye progresivamente con el desgaste. El principal limitante es el dataset de solo 3 pacientes.



\*\*Problema 2:\*\* La CNN baseline alcanzó MAE=10.21 años cumpliendo el enunciado. Como iniciativa adicional, Transfer Learning con ResNet18 superó significativamente ese resultado — R² de 0.369 a 0.664 y MAE de 10.21 a 7.58 años. Con 50k+ imágenes y fine-tuning completo se podría alcanzar MAE < 5 años.

