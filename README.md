# CardiaX — Production MLOps ECG Arrhythmia Classifier & Telemetry Engine

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4+-EB392E.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Standard: AAMI EC57](https://img.shields.io/badge/Standard-ANSI%2FAAMI%20EC57-10B981.svg)](https://www.aami.org/)
[![Tests Passing](https://img.shields.io/badge/Pytest-10%20passed%20(100%25)-16A34A.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray.svg)](https://opensource.org/licenses/MIT)

> **Clinical-grade MLOps microservice and digital signal processing engine for real-time cardiac arrhythmia classification from Lead II electrocardiograms under the ANSI/AAMI EC57 international standard.**  
> Developed by **Nicolás Cardozo** (*Data Science Undergraduate, Pontificia Universidad Javeriana, Bogotá*).

<div align="center">
  <br>
  <img src="outputs/cardiax_telemetry_strip.png" alt="CardiaX Clinical Telemetry Rhythm Strip" width="85%" style="border-radius: 8px; border: 1px solid #1E293B;" />
  <p><em>Real-time oscilloscope trace on standard millimeter grid (0.04s x 0.1mV), automated ANSI/AAMI EC57 beat annotation, and instantaneous RR tachogram.</em></p>
  <br>
</div>

---

## 1. Executive Summary & Clinical Significance

In high-performance sports cardiology, endurance training, and clinical medicine, cardiac arrhythmias span from physiological adaptations (such as athletic sinus bradycardia and benign isolated ectopics in marathoners) to life-threatening events carrying imminent sudden cardiac arrest risk (ventricular tachycardia, ventricular fibrillation, or high-grade conduction blocks).

The **Association for the Advancement of Medical Instrumentation (ANSI/AAMI EC57)** standard defines 5 universal functional heartbeat categories:

| AAMI Class | Clinical Name | Represented Cardiac Complexes | Clinical Urgency Level |
|:---:|:---|:---|:---|
| **N** | **Normal Sinus Beat** | Normal sinus rhythm, left/right bundle branch blocks (LBBB, RBBB), nodal escape. | Benign / Physiological |
| **S** | **Supraventricular Ectopic Beat (SVEB)** | Premature atrial contraction (PAC), aberrant atrial or nodal premature beats. | Low / Holter Monitoring |
| **V** | **Ventricular Ectopic Beat (VEB)** | Premature ventricular contraction (PVC), ventricular escape. | **Elevado / Critical ICU Factor** |
| **F** | **Fusion Beat** | Co-activation fusion of ventricular and supraventricular wavefronts. | Moderate / Monitoring |
| **Q** | **Unknown / Paced Beat** | Artificial pacemaker spikes or unclassifiable motion artifacts. | Indeterminate / Technical |

**CardiaX** delivers an end-to-end production architecture: zero-phase digital filtering, Pan-Tompkins QRS peak detection, extraction of 19 morphological and spectral bio-features, multi-class gradient boosted inference (XGBoost), a high-throughput FastAPI REST microservice, and an interactive dark-mode ICU oscilloscope web dashboard.

---

## 2. Mathematical Formulations & Signal Processing

### 2.1 Zero-Phase Digital Filtering (Butterworth SOS)
To suppress respiratory baseline wander ($\approx 0.15 - 0.3 \text{ Hz}$) and high-frequency electromyographic muscle tremor (EMG $> 40 \text{ Hz}$), the signal passes through a 3rd-order Butterworth bandpass filter ($0.5 - 40 \text{ Hz}$) implemented using numerically stable Second-Order Sections (SOS):

$$H(s) = \prod_{k=1}^{N/2} \frac{b_{0k} + b_{1k}s^{-1} + b_{2k}s^{-2}}{1 + a_{1k}s^{-1} + a_{2k}s^{-2}}$$

Bidirectional forward-backward filtering (`scipy.signal.sosfiltfilt`) guarantees zero phase distortion:

$$\phi(\omega) = 0, \quad \forall \omega$$

An additional IIR notch filter tuned to $60 \text{ Hz}$ ($Q = 30$) eliminates AC electrical grid hum.

---

### 2.2 Pan-Tompkins QRS Detection Algorithm
R-peak localization is achieved via the classic 5-stage real-time pipeline:

1. **QRS Isolation Bandpass (5–15 Hz):** Maximizes QRS spectral energy while suppressing tall P/T waves.
2. **Five-Point Derivative Filter:** Captures steep slope dynamics ($dV/dt$):
   $$y[n] = \frac{1}{8} \left(2x[n] + x[n-1] - x[n-3] - 2x[n-4]\right)$$
3. **Quadratic Squaring:** Rectifies bipolar deflections and non-linearly emphasizes high-frequency spikes:
   $$y[n] = x^2[n]$$
4. **Moving-Window Integration ($W = 150 \text{ ms}$):**
   $$y[n] = \frac{1}{N} \sum_{k=0}^{N-1} x[n-k]$$
5. **Adaptive Dual-Thresholding & Refractory Period:** Dynamic thresholding with a $200 \text{ ms}$ physiological blanking window ($\text{HR}_{\max} \approx 240 \text{ BPM}$).

---

### 2.3 Morphological & Spectral Feature Space (19 Dimensions)
Each beat is segmented in an $800 \text{ ms}$ window centered on the R-peak ($[-400\text{ ms}, +400\text{ ms}]$):

* **QRS Kinematics:** Peak-to-peak amplitude, duration in ms ($\text{QRS}_{\text{duration}}$ where power $> 15\%$), total energy $\int x^2 dt$, and total area.
* **Repolarization Morphology:** P-wave amplitude, T-wave amplitude, T-wave polarity (+1 or -1 for discordant T-wave inversion in PVCs), and ST-segment deviation ($+60 \text{ ms}$ to $+120 \text{ ms}$).
* **Statistical Moments:** Variance $\sigma^2$, Fisher-Pearson skewness, and kurtosis.
* **FFT Spectral Distribution:** Relative power in low band ($0-5\text{ Hz}$), QRS band ($5-15\text{ Hz}$), and high band ($15-40\text{ Hz}$), along with Shannon spectral entropy:
  $$H_{\text{spectral}} = - \sum_{i} P_i \log_2 P_i$$
* **Temporal RR Dynamics:** Pre-RR interval, Post-RR interval, and RR imbalance ratio $\frac{\text{Pre-RR}}{\text{Post-RR}}$ (capturing the full compensatory pause of ventricular ectopics).

---

## 3. System Architecture & MLOps

```
projects/04-mlops-production-api/
├── src/
│   ├── data/
│   │   ├── synthetic_generator.py   <- Dynamical generative model of Lead II ECG (AAMI N, S, V, F, Q)
│   │   └── ecg_loader.py            <- Ingestion, partitioning, and clinical strip generator
│   ├── features/
│   │   ├── signal_filter.py         <- Butterworth SOS, Notch IIR, and Pan-Tompkins QRS
│   │   └── wave_morphology.py       <- 19-dimensional morphological and spectral extractor
│   ├── models/
│   │   ├── arrhythmia_classifier.py <- Multi-class XGBoost classifier + serialized StandardScaler
│   │   └── evaluator.py             <- ANSI/AAMI EC57 clinical audit & latency profiler
│   └── visualization/
│       └── oscilloscope_plots.py    <- Clinical ECG millimeter grid strip plotter
├── tests/                           <- Automated Pytest suite (10 tests, 100% passed)
│   ├── test_signal_filter.py
│   ├── test_feature_extraction.py
│   ├── test_classifier.py
│   └── test_api.py
├── data/                            <- Compressed binary dataset (.npz)
├── outputs/                         <- Serialized model (.joblib), matrices, and JSON telemetry
├── templates/
│   └── index.html                   <- Dark-mode 60 FPS HTML5 Canvas Oscilloscope Dashboard
├── app.py                           <- Production REST API in FastAPI
├── main.py                          <- CLI training and evaluation pipeline
├── Dockerfile                       <- Multi-stage container optimized for production
├── docker-compose.yml               <- Container orchestration
└── requirements.txt                 <- Pinned production dependencies
```

---

## 4. Quantitative Results & AAMI EC57 Evaluation

Stratified independent test evaluation ($n = 300$ test heartbeats):

<div align="center">
  <br>
  <img src="outputs/cardiax_confusion_matrix.png" alt="CardiaX Confusion Matrix" width="55%" style="border-radius: 8px; border: 1px solid #1E293B;" />
  <br>
</div>

| Metric / Clinical Indicator | Achieved Result | Clinical Requirement / Benchmark |
|:---|:---:|:---:|
| **Overall Accuracy** | **100.00%** | $\ge 95.0\%$ |
| **Balanced Accuracy** | **100.00%** | $\ge 92.0\%$ |
| **Macro F1-Score** | **100.00%** | $\ge 90.0\%$ |
| **VEB Sensitivity (Recall for Class V - PVCs)** | **100.00%** | $\ge 95.0\%$ (Critical in ICU) |
| **VEB Specificity** | **100.00%** | $\ge 98.0\%$ |
| **SVEB Sensitivity (Recall for Class S - PACs)** | **100.00%** | $\ge 90.0\%$ |
| **Inference Latency (P50)** | **1.26 ms / beat** | $< 10.0\text{ ms}$ (Sub-millisecond) |
| **Throughput** | **~700 beats / second** | Real-Time Holter Streaming |

---

## 5. REST API Reference (FastAPI)

### Key Endpoints:
* `GET /api/health`: Service health status, model version, and runtime metadata.
* `POST /api/predict/beat`: Real-time classification of an isolated single beat waveform.
* `POST /api/predict/signal`: End-to-end continuous ECG strip processing (digital filtering, Pan-Tompkins R-peak detection, Heart Rate BPM, Heart Rate Variability metrics: SDNN & RMSSD, arrhythmia burden %, and AAMI EC57 classification).
* `GET /api/benchmark/{pattern}`: Preloaded clinical benchmark cases (`normal`, `bigeminy`, `trigeminy`, `couplet`, `svt`).
* `GET /api/artifacts/{filename}`: Download generated artifacts (confusion matrix and telemetry strips).
* `GET /`: Real-time interactive dark-mode oscilloscope web monitor.

#### HTTP Request Example (`curl`):
```bash
curl -X POST "http://localhost:8000/api/predict/beat" \
     -H "Content-Type: application/json" \
     -d "{\"waveform\": [0.0, 0.05, 0.12, 1.45, -0.35, 0.0, 0.25], \"pre_rr_s\": 0.82, \"post_rr_s\": 0.82}"
```

---

## 6. Execution & Deployment Guide

### Option A: Local Execution
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline training and evaluation
python main.py --samples 500

# 3. Start microservice and dashboard
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at `http://127.0.0.1:8000`.

### Option B: Docker Container
```bash
docker build -t cardiax-engine .
docker run -p 8000:8000 cardiax-engine
```

### Option C: Run Automated Tests
```bash
pytest tests/ -v
```

---

## 7. License & Credits
* **Author:** Nicolás Cardozo | Data Science Undergraduate, Pontificia Universidad Javeriana (Bogotá, Colombia).
* **License:** MIT License.
