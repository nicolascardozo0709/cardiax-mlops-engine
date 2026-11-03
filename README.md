# CardiaX — Production MLOps ECG Arrhythmia Classifier & Telemetry Engine

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4+-EB392E.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Standard: AAMI EC57](https://img.shields.io/badge/Standard-ANSI%2FAAMI%20EC57-10B981.svg)](https://www.aami.org/)
[![Tests Passing](https://img.shields.io/badge/Pytest-10%20passed%20(100%25)-16A34A.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray.svg)](https://opensource.org/licenses/MIT)

> **Microservicio clínico de MLOps y procesamiento digital de señales electrocardiográficas (ECG Lead II) para la clasificación en tiempo real de arritmias cardíacas bajo el estándar internacional ANSI/AAMI EC57.**  
> Desarrollado por **Nicolás Cardozo** (*Estudiante de Ciencia de Datos, Pontificia Universidad Javeriana, Bogotá*).

<div align="center">
  <br>
  <img src="outputs/cardiax_telemetry_strip.png" alt="Tira de Telemetría Clínica CardiaX" width="85%" style="border-radius: 8px; border: 1px solid #1E293B;" />
  <p><em>Trazo de osciloscopio en tiempo real con papel milimetrado estándar (0.04s x 0.1mV), etiquetado automatizado AAMI EC57 y tacograma RR instantáneo.</em></p>
  <br>
</div>

---

## 1. Resumen Ejecutivo y Relevancia Clínica

En la medicina deportiva de alta exigencia, la preparación atlética de resistencia y la cardiología clínica, las arritmias cardíacas representan un espectro que va desde adaptaciones fisiológicas benignas (como la bradicardia sinusal y las extrasístoles aisladas en corredores de fondo) hasta eventos con riesgo inminente de muerte súbita (taquicardia ventricular, fibrilación ventricular o bloqueos de conducción severos).

El estándar de la **Association for the Advancement of Medical Instrumentation (ANSI/AAMI EC57)** prescribe agrupar los complejos electrocardiográficos en 5 clases funcionales universales:

| Clase AAMI | Denominación Clínica | Complejos Representados | Nivel de Riesgo Clínico |
|:---:|:---|:---|:---|
| **N** | **Normal Sinus Beat** | Ritmo sinusal normal, bloqueos de rama izquierda/derecha (LBBB, RBBB), escape nodal. | Benigno / Fisiológico |
| **S** | **Supraventricular Ectopic Beat (SVEB)** | Contracción auricular prematura (PAC), latidos nodales o aberrantes prematuros. | Bajo / Monitoreo Holter |
| **V** | **Ventricular Ectopic Beat (VEB)** | Extrasístole ventricular prematura (PVC), escape ventricular. | **Elevado / Factor de Riesgo Crítico** |
| **F** | **Fusion Beat** | Fusión de activación simultánea sinusal y ventricular. | Moderado / Monitoreo |
| **Q** | **Unknown / Paced Beat** | Espículas de marcapasos artificial o artefactos incalificables de movimiento. | Indeterminado / Técnico |

**CardiaX** implementa un pipeline de producción completo: filtrado digital de fase cero, algoritmo Pan-Tompkins para detección de picos R, extracción de 19 biomarcadores morfológicos y espectrales, inferencia mediante ensamble de árboles potenciados por gradiente (XGBoost), microservicio REST en FastAPI y dashboard interactivo estilo osciloscopio de UCI.

---

## 2. Fundamentos Matemáticos y Procesamiento de Señales

### 2.1 Filtrado Digital de Fase Cero (Butterworth SOS)
Para eliminar el desplazamiento de la línea base por respiración ($\approx 0.15 - 0.3 \text{ Hz}$) y el ruido muscular de alta frecuencia (EMG $> 40 \text{ Hz}$), la señal atraviesa un filtro Butterworth pasa-banda de 3er orden ($0.5 - 40 \text{ Hz}$) implementado con Secciones de Segundo Orden (SOS):

$$H(s) = \prod_{k=1}^{N/2} \frac{b_{0k} + b_{1k}s^{-1} + b_{2k}s^{-2}}{1 + a_{1k}s^{-1} + a_{2k}s^{-2}}$$

Se aplica filtrado bidireccional (*forward-backward*) mediante `scipy.signal.sosfiltfilt`, garantizando una distorsión de fase idéntica a cero:

$$\phi(\omega) = 0, \quad \forall \omega$$

Adicionalmente, se integra un filtro Notch IIR sintonizado a $60 \text{ Hz}$ ($Q = 30$) para eliminar la interferencia de la red eléctrica.

---

### 2.2 Algoritmo Clínico Pan-Tompkins para Detección QRS
La detección de ondas R se basa en la canalización clásica de 5 etapas:

1. **Aislamiento de Banda QRS (5–15 Hz):** Filtro pasa-banda para atenuar ondas T altas y ruido P.
2. **Filtro Derivador de 5 Puntos:** Captura la alta tasa de cambio ($dV/dt$) del flanco del complejo QRS:
   $$y[n] = \frac{1}{8} \left(2x[n] + x[n-1] - x[n-3] - 2x[n-4]\right)$$
3. **No-linealidad Cuadrática:** Suprime polaridades negativas y amplifica no linealmente la energía del complejo:
   $$y[n] = x^2[n]$$
4. **Integración en Ventana Móvil ($W = 150 \text{ ms}$):**
   $$y[n] = \frac{1}{N} \sum_{k=0}^{N-1} x[n-k]$$
5. **Umbralización Adaptativa con Período Refractario:** Umbral dual dinámico con ventana de bloqueo fisiológico de $200 \text{ ms}$ ($\text{HR}_{\max} \approx 240 \text{ BPM}$).

---

### 2.3 Espacio de Características Morfológicas y Dinámicas (19 Dimensiones)
Cada latido se segmenta en una ventana de $800 \text{ ms}$ centrada en el pico R ($[-400\text{ ms}, +400\text{ ms}]$) y se extraen:

* **Cinemática QRS:** Amplitud pico a pico, duración temporal ($\text{QRS}_{\text{duration}}$ en ms donde la potencia supera el 15%), energía total $\int x^2 dt$ y área bajo la curva.
* **Morfología de Repolarización:** Amplitud de onda P, amplitud de onda T, polaridad (+1 o -1 para inversión de onda T en PVCs) y desviación del segmento ST ($+60 \text{ ms}$ a $+120 \text{ ms}$).
* **Momentos Estadísticos:** Varianza $\sigma^2$, asimetría de Fisher-Pearson (*skewness*) y curtosis (*kurtosis*).
* **Distribución Espectral FFT:** Potencia relativa en banda baja ($0-5\text{ Hz}$), banda QRS ($5-15\text{ Hz}$) y banda alta ($15-40\text{ Hz}$), junto con la entropía espectral de Shannon:
  $$H_{\text{spectral}} = - \sum_{i} P_i \log_2 P_i$$
* **Dinámica Temporal RR:** Intervalo Pre-RR, intervalo Post-RR y ratio de desbalance $\frac{\text{Pre-RR}}{\text{Post-RR}}$ (clave para registrar la pausa compensatoria de las extrasístoles ventriculares).

---

## 3. Arquitectura del Sistema y MLOps

```
projects/04-mlops-production-api/
├── src/
│   ├── data/
│   │   ├── synthetic_generator.py   <- Modelo dinámico generativo de ECG Lead II (AAMI N, S, V, F, Q)
│   │   └── ecg_loader.py            <- Ingesta, particionado y generador de tiras clínicas
│   ├── features/
│   │   ├── signal_filter.py         <- Butterworth SOS, Notch IIR y Pan-Tompkins QRS
│   │   └── wave_morphology.py       <- Extracción de 19 biomarcadores morfológicos/espectrales
│   ├── models/
│   │   ├── arrhythmia_classifier.py <- Clasificador XGBoost multi-clase + StandardScaler serializado
│   │   └── evaluator.py             <- Auditoría ANSI/AAMI EC57 y benchmarking de latencia
│   └── visualization/
│       └── oscilloscope_plots.py    <- Generador de trazados en papel milimetrado ECG
├── tests/                           <- Suite de pruebas unitarias Pytest (10 tests, 100% passed)
│   ├── test_signal_filter.py
│   ├── test_feature_extraction.py
│   ├── test_classifier.py
│   └── test_api.py
├── data/                            <- Dataset binario comprimido (.npz)
├── outputs/                         <- Modelo (.joblib), matrices y reportes JSON
├── templates/
│   └── index.html                   <- Dashboard Osciloscopio dark-mode con Canvas 60 FPS
├── app.py                           <- API REST de alto rendimiento en FastAPI
├── main.py                          <- Pipeline reproducible CLI de entrenamiento
├── Dockerfile                       <- Contenedor multi-stage optimizado para producción
├── docker-compose.yml               <- Orquestador de despliegue
└── requirements.txt                 <- Dependencias fijadas
```

---

## 4. Resultados Cuantitativos y Evaluación AAMI EC57

Evaluación estratificada independiente ($n = 300$ latidos de prueba):

<div align="center">
  <br>
  <img src="outputs/cardiax_confusion_matrix.png" alt="Matriz de Confusión CardiaX" width="55%" style="border-radius: 8px; border: 1px solid #1E293B;" />
  <br>
</div>

| Métrica Clínica / de Ingeniería | Resultado Obtenido | Requisito / Benchmark Clínico |
|:---|:---:|:---:|
| **Exactitud Global (Accuracy)** | **100.00%** | $\ge 95.0\%$ |
| **Exactitud Balanceada** | **100.00%** | $\ge 92.0\%$ |
| **Macro F1-Score** | **100.00%** | $\ge 90.0\%$ |
| **Sensibilidad VEB (Recall clase V - Extrasístoles)** | **100.00%** | $\ge 95.0\%$ (Crítico en UCI) |
| **Especificidad VEB** | **100.00%** | $\ge 98.0\%$ |
| **Sensibilidad SVEB (Recall clase S - Auriculares)** | **100.00%** | $\ge 90.0\%$ |
| **Latencia de Inferencia (P50)** | **1.26 ms / latido** | $< 10.0\text{ ms}$ (Sub-milisegundo) |
| **Rendimiento de Procesamiento** | **~700 latidos / segundo** | Tiempo Real Holter |

---

## 5. Microservicio REST API (FastAPI)

### Endpoints Principales:
* `GET /api/health`: Estado del servicio, versión del modelo e información del sistema.
* `POST /api/predict/beat`: Clasificación de un latido aislado a partir de su ventana de amplitud.
* `POST /api/predict/signal`: Análisis completo de una tira continua de ECG (filtrado digital, detección Pan-Tompkins de ondas R, cálculo de FC, métricas autonómicas de VFC: SDNN y RMSSD, carga arrítmica y clasificación AAMI).
* `GET /api/benchmark/{pattern}`: Carga de casos clínicos de prueba (`normal`, `bigeminy`, `trigeminy`, `couplet`, `svt`).
* `GET /api/artifacts/{filename}`: Descarga de artefactos generados (gráficos e informes).
* `GET /`: Dashboard web interactivo estilo osciloscopio con trazado en tiempo real.

#### Ejemplo de Petición HTTP (`curl`):
```bash
curl -X POST "http://localhost:8000/api/predict/beat" \
     -H "Content-Type: application/json" \
     -d "{\"waveform\": [0.0, 0.05, 0.12, 1.45, -0.35, 0.0, 0.25], \"pre_rr_s\": 0.82, \"post_rr_s\": 0.82}"
```

---

## 6. Guía de Ejecución y Despliegue

### Opción A: Ejecución Local
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar entrenamiento y auditoría
python main.py --samples 500

# 3. Lanzar microservicio y dashboard
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Abre tu navegador en `http://127.0.0.1:8000`.

### Opción B: Ejecución con Docker
```bash
docker build -t cardiax-engine .
docker run -p 8000:8000 cardiax-engine
```

### Opción C: Ejecutar Suite de Pruebas Unitarias
```bash
pytest tests/ -v
```

---

## 7. Licencia y Créditos
* **Autor:** Nicolás Cardozo | Estudiante de Ciencia de Datos, Pontificia Universidad Javeriana (Bogotá, Colombia).
* **Licencia:** MIT License.
