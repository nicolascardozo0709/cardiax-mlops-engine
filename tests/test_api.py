"""Unit tests for CardiaX FastAPI REST API endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from src.data.synthetic_generator import SyntheticECGGenerator

client = TestClient(app)


def test_api_health():
    """Verify health endpoint returns status 200 and metadata."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "CardiaX Arrhythmia Engine"
    assert data["model_loaded"] is True


def test_api_predict_beat():
    """Verify single beat classification endpoint."""
    generator = SyntheticECGGenerator(sampling_rate_hz=360, random_state=42)
    beat, _ = generator.generate_single_beat("N", window_duration_s=0.8)

    payload = {
        "waveform": beat.tolist(),
        "pre_rr_s": 0.83,
        "post_rr_s": 0.83,
    }
    res = client.post("/api/predict/beat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] in ["N", "S", "V", "F", "Q"]
    assert "inference_latency_ms" in data
    assert data["inference_latency_ms"] >= 0


def test_api_predict_signal():
    """Verify continuous signal analysis endpoint."""
    generator = SyntheticECGGenerator(sampling_rate_hz=360, random_state=42)
    strip = generator.generate_rhythm_strip(duration_s=4.0, heart_rate_bpm=75.0, arrhythmia_pattern="normal")

    payload = {
        "signal": strip["raw_signal"].tolist(),
        "sampling_rate_hz": 360,
        "apply_notch_filter": True,
        "patient_id": "TEST-PT-01",
    }
    res = client.post("/api/predict/signal", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_beats_detected"] >= 3
    assert data["mean_heart_rate_bpm"] > 50
    assert "filtered_signal_subsample" in data


def test_api_benchmark_endpoint():
    """Verify clinical benchmark endpoint."""
    res = client.get("/api/benchmark/bigeminy")
    assert res.status_code == 200
    data = res.json()
    assert data["patient_id"] == "CASE-BIGEMINY"
    assert data["total_beats_detected"] > 0


def test_index_view():
    """Verify HTML dashboard loads successfully."""
    res = client.get("/")
    assert res.status_code == 200
    assert "CardiaX" in res.text
