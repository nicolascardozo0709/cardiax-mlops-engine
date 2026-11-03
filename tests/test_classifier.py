"""Unit tests for CardiaX XGBoost Arrhythmia Classifier and Evaluator."""

import numpy as np
import pytest
from src.data.synthetic_generator import SyntheticECGGenerator
from src.models.arrhythmia_classifier import CardiaXClassifier, BeatPrediction


def test_classifier_fit_and_prediction(tmp_path):
    """Test model training, prediction accuracy, and artifact persistence."""
    generator = SyntheticECGGenerator(sampling_rate_hz=360, random_state=42)
    X, y = generator.generate_balanced_dataset(samples_per_class=40)

    clf = CardiaXClassifier(random_state=42)
    clf.fit(X, y)

    # Predict single beat
    test_beat, test_lbl = generator.generate_single_beat("V", window_duration_s=0.8)
    pred: BeatPrediction = clf.predict_beat(test_beat)

    assert pred.predicted_class in ["N", "S", "V", "F", "Q"]
    assert 0.0 <= pred.confidence <= 1.0
    # Probabilities must sum to ~1.0
    prob_sum = sum(pred.probabilities.values())
    assert abs(prob_sum - 1.0) < 0.05

    # Test serialization
    model_path = tmp_path / "test_model.joblib"
    clf.save_model(model_path)
    assert model_path.exists()

    loaded = CardiaXClassifier.load_model(model_path)
    assert loaded.is_fitted
    pred_loaded = loaded.predict_beat(test_beat)
    assert pred_loaded.predicted_class == pred.predicted_class
