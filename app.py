"""CardiaX Production MLOps REST API & Medical Telemetry Microservice.

FastAPI service providing real-time arrhythmia classification, continuous
ECG rhythm strip analysis, Pan-Tompkins QRS peak detection, HRV metrics,
and interactive oscilloscope dashboard serving.
"""

from __future__ import annotations
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.data.ecg_loader import get_clinical_benchmark_strip
from src.features.signal_filter import ECGFilter, PanTompkinsQRSDetector
from src.models.arrhythmia_classifier import CardiaXClassifier, CLASS_DESCRIPTIONS, CLINICAL_RISK_LEVEL

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

MODEL_FILE = OUTPUTS_DIR / "cardiax_model.joblib"

classifier_instance: Optional[CardiaXClassifier] = None


def get_classifier() -> Optional[CardiaXClassifier]:
    """Get active classifier instance, loading from disk if not yet loaded."""
    global classifier_instance
    if classifier_instance is None and MODEL_FILE.exists():
        try:
            classifier_instance = CardiaXClassifier.load_model(MODEL_FILE)
        except Exception as e:
            print(f"[WARN] Failed lazy loading model: {e}")
    return classifier_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook: preload trained CardiaX model into memory on startup."""
    global classifier_instance
    print("[INIT] Booting CardiaX MLOps Arrhythmia Engine...")
    if MODEL_FILE.exists():
        classifier_instance = CardiaXClassifier.load_model(MODEL_FILE)
        print(f"[OK] Preloaded CardiaX model version {classifier_instance.version}")
    else:
        print("[WARN] Model file not found at startup. Running quick self-training...")
        from main import run_pipeline
        run_pipeline(samples_per_class=300, output_dir=OUTPUTS_DIR)
        classifier_instance = CardiaXClassifier.load_model(MODEL_FILE)
    yield
    print("[SHUTDOWN] CardiaX Microservice stopping.")


app = FastAPI(
    title="CardiaX — Clinical ECG Arrhythmia MLOps Engine",
    description="ANSI/AAMI EC57 compliant arrhythmia classifier, Pan-Tompkins QRS detector, and telemetry API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------------------------
class SingleBeatRequest(BaseModel):
    waveform: List[float] = Field(..., description="ECG amplitude values centered near R-peak (~288 samples for 0.8s @ 360Hz)")
    pre_rr_s: Optional[float] = Field(None, description="Preceding RR interval in seconds")
    post_rr_s: Optional[float] = Field(None, description="Succeeding RR interval in seconds")


class SingleBeatResponse(BaseModel):
    predicted_class: str
    class_name: str
    confidence: float
    clinical_risk: str
    probabilities: Dict[str, float]
    features: Dict[str, float]
    inference_latency_ms: float


class SignalAnalysisRequest(BaseModel):
    signal: List[float] = Field(..., description="Continuous raw ECG lead II signal")
    sampling_rate_hz: int = Field(360, ge=50, le=2000, description="Sampling rate in Hz")
    apply_notch_filter: bool = Field(True, description="Whether to apply 60Hz powerline notch filter")
    patient_id: Optional[str] = Field("ANON-001", description="Patient identifier")


class BeatAnalysisItem(BaseModel):
    peak_sample: int
    peak_time_s: float
    predicted_class: str
    class_name: str
    confidence: float
    clinical_risk: str
    probabilities: Dict[str, float]


class SignalAnalysisResponse(BaseModel):
    patient_id: str
    duration_s: float
    total_beats_detected: int
    mean_heart_rate_bpm: float
    rr_intervals_ms: List[float]
    sdnn_ms: float
    rmssd_ms: float
    arrhythmia_burden_percent: float
    class_distribution: Dict[str, int]
    clinical_alert: str
    detected_beats: List[BeatAnalysisItem]
    filtered_signal_subsample: List[float]
    time_subsample: List[float]
    total_processing_time_ms: float


# ----------------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Health status and telemetry endpoint."""
    clf = get_classifier()
    return {
        "status": "healthy",
        "service": "CardiaX Arrhythmia Engine",
        "standard": "ANSI/AAMI EC57",
        "model_loaded": clf is not None,
        "model_version": clf.version if clf else None,
        "developer": "Nicolas Cardozo",
        "institution": "Pontificia Universidad Javeriana",
    }


@app.post("/api/predict/beat", response_model=SingleBeatResponse)
def predict_single_beat(payload: SingleBeatRequest):
    """Classify a single isolated heartbeat waveform into AAMI EC57 categories."""
    clf = get_classifier()
    if clf is None:
        raise HTTPException(status_code=503, detail="CardiaX model not initialized.")

    t0 = time.perf_counter()
    wf_arr = np.asarray(payload.waveform, dtype=float)
    pred = clf.predict_beat(
        wf_arr, pre_rr_s=payload.pre_rr_s, post_rr_s=payload.post_rr_s
    )
    t1 = time.perf_counter()
    lat_ms = (t1 - t0) * 1000.0

    return SingleBeatResponse(
        predicted_class=pred.predicted_class,
        class_name=pred.class_name,
        confidence=pred.confidence,
        clinical_risk=pred.clinical_risk,
        probabilities=pred.probabilities,
        features=pred.features,
        inference_latency_ms=round(lat_ms, 3),
    )


@app.post("/api/predict/signal", response_model=SignalAnalysisResponse)
def analyze_continuous_signal(payload: SignalAnalysisRequest):
    """Perform full-strip clinical processing: filtering, Pan-Tompkins QRS detection,

    per-beat AAMI classification, and HRV biometric computation.
    """
    clf = get_classifier()
    if clf is None:
        raise HTTPException(status_code=503, detail="CardiaX model not initialized.")

    t_start = time.perf_counter()
    raw_sig = np.asarray(payload.signal, dtype=float)
    fs = payload.sampling_rate_hz
    n_samples = len(raw_sig)
    duration_s = n_samples / fs

    if duration_s < 1.0:
        raise HTTPException(status_code=400, detail="Signal must be at least 1.0 second in duration.")

    # 1. Digital Filter Suite (0.5-40Hz Bandpass + 60Hz Notch)
    ecg_filter = ECGFilter(sampling_rate_hz=fs)
    filtered = ecg_filter.butter_bandpass(raw_sig, lowcut=0.5, highcut=40.0)
    if payload.apply_notch_filter:
        filtered = ecg_filter.notch_filter(filtered, notch_freq=60.0)

    # 2. Pan-Tompkins QRS Complex Detection
    qrs_detector = PanTompkinsQRSDetector(sampling_rate_hz=fs)
    peaks = qrs_detector.detect_peaks(filtered)

    # 3. Heart Rate and HRV Computation (SDNN, RMSSD)
    rr_intervals_s = []
    for i in range(1, len(peaks)):
        rr_intervals_s.append((peaks[i] - peaks[i - 1]) / fs)

    rr_ms = [round(r * 1000.0, 1) for r in rr_intervals_s]

    if rr_ms:
        mean_bpm = float(round(60000.0 / np.mean(rr_ms), 1))
        sdnn = float(round(np.std(rr_ms), 2))
        rmssd = float(round(np.sqrt(np.mean(np.diff(rr_ms) ** 2)), 2)) if len(rr_ms) > 1 else 0.0
    else:
        mean_bpm = 0.0
        sdnn = 0.0
        rmssd = 0.0

    # 4. Slicing and Multi-Class Classification
    half_w = int(0.400 * fs)  # 400ms half-window
    detected_beats: List[BeatAnalysisItem] = []
    class_dist = {"N": 0, "S": 0, "V": 0, "F": 0, "Q": 0}

    for i, pk in enumerate(peaks):
        if pk - half_w >= 0 and pk + half_w <= n_samples:
            beat_wf = filtered[pk - half_w : pk + half_w]
            pre_rr = rr_intervals_s[i - 1] if i > 0 and i - 1 < len(rr_intervals_s) else None
            post_rr = rr_intervals_s[i] if i < len(rr_intervals_s) else None
            pred = clf.predict_beat(beat_wf, pre_rr_s=pre_rr, post_rr_s=post_rr)
            cls_lbl = pred.predicted_class
        else:
            cls_lbl = "N"
            pred = None

        class_dist[cls_lbl] = class_dist.get(cls_lbl, 0) + 1

        detected_beats.append(
            BeatAnalysisItem(
                peak_sample=pk,
                peak_time_s=round(pk / fs, 3),
                predicted_class=cls_lbl,
                class_name=CLASS_DESCRIPTIONS.get(cls_lbl, "Normal Sinus"),
                confidence=pred.confidence if pred else 0.99,
                clinical_risk=CLINICAL_RISK_LEVEL.get(cls_lbl, "BENIGN / NORMAL"),
                probabilities=pred.probabilities if pred else {"N": 1.0, "S": 0.0, "V": 0.0, "F": 0.0, "Q": 0.0},
            )
        )

    # Arrhythmia burden
    total_beats = len(detected_beats)
    non_normal_beats = total_beats - class_dist.get("N", 0)
    burden = round((non_normal_beats / max(1, total_beats)) * 100.0, 1)

    # Clinical Alert Assessment
    if class_dist.get("V", 0) >= 3 or burden > 30.0:
        alert = "CRITICAL: FREQUENT VENTRICULAR ECTOPY DETECTED"
    elif class_dist.get("V", 0) > 0:
        alert = "ELEVATED: ISOLATED PVC DETECTED"
    elif class_dist.get("S", 0) > 0:
        alert = "MONITOR: SUPRAVENTRICULAR ECTOPIC ACTIVITY"
    else:
        alert = "NORMAL: REGULAR SINUS RHYTHM"

    # Subsample signal for UI transmission (max 2000 points)
    step = max(1, n_samples // 1500)
    sub_sig = [round(float(v), 3) for v in filtered[::step]]
    sub_time = [round(float(i / fs), 3) for i in range(0, n_samples, step)]

    t_end = time.perf_counter()
    proc_ms = (t_end - t_start) * 1000.0

    return SignalAnalysisResponse(
        patient_id=payload.patient_id or "ANON-001",
        duration_s=round(duration_s, 2),
        total_beats_detected=total_beats,
        mean_heart_rate_bpm=mean_bpm,
        rr_intervals_ms=rr_ms,
        sdnn_ms=sdnn,
        rmssd_ms=rmssd,
        arrhythmia_burden_percent=burden,
        class_distribution=class_dist,
        clinical_alert=alert,
        detected_beats=detected_beats,
        filtered_signal_subsample=sub_sig,
        time_subsample=sub_time,
        total_processing_time_ms=round(proc_ms, 2),
    )


@app.get("/api/benchmark/{pattern}")
def get_benchmark(pattern: str = "bigeminy"):
    """Fetch pre-synthesized clinical benchmark case for live UI demonstration."""
    if pattern not in ["normal", "bigeminy", "trigeminy", "couplet", "svt", "mixed"]:
        pattern = "bigeminy"

    strip = get_clinical_benchmark_strip(pattern=pattern)
    # Automatically run analysis on the strip
    req = SignalAnalysisRequest(
        signal=strip["raw_signal"].tolist(),
        sampling_rate_hz=360,
        apply_notch_filter=True,
        patient_id=f"CASE-{pattern.upper()}",
    )
    return analyze_continuous_signal(req)


@app.get("/api/artifacts/{filename}")
def get_artifact(filename: str):
    """Download generated artifact files."""
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(path=file_path)


@app.get("/", response_class=HTMLResponse)
def index_view():
    """Serve the clinical telemetry oscilloscope UI."""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found.")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
