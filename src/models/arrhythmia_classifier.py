"""CardiaX Production Arrhythmia Classifier.

Ensemble Gradient Boosted Engine for multi-class ECG heartbeat classification
adhering to the ANSI/AAMI EC57 standard:
- N: Normal Sinus Beat
- S: Supraventricular Ectopic Beat (PAC)
- V: Ventricular Ectopic Beat (PVC)
- F: Fusion Beat
- Q: Paced / Unknown Beat
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.wave_morphology import HeartbeatFeatureExtractor, FEATURE_NAMES


CLASS_MAPPING = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}
INV_CLASS_MAPPING = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}

CLASS_DESCRIPTIONS = {
    "N": "Normal Sinus Depolarization",
    "S": "Supraventricular Ectopic Beat (PAC / SVEB)",
    "V": "Ventricular Ectopic Beat (PVC / VEB - Risk Factor)",
    "F": "Fusion Beat (Ventricular + Sinus Co-Activation)",
    "Q": "Paced / Unclassified / Artifact Beat",
}

CLINICAL_RISK_LEVEL = {
    "N": "BENIGN / NORMAL",
    "S": "LOW / MONITOR",
    "V": "ELEVATED / CLINICAL REVIEW",
    "F": "MODERATE / MONITOR",
    "Q": "INDETERMINATE / ARTIFACT",
}


@dataclass
class BeatPrediction:
    """Structured clinical prediction for an individual heartbeat."""
    predicted_class: str
    class_name: str
    confidence: float
    clinical_risk: str
    probabilities: Dict[str, float]
    features: Dict[str, float]

    def to_dict(self) -> Dict[str, any]:
        return asdict(self)


class CardiaXClassifier:
    """Production Machine Learning Classifier for ECG Arrhythmias."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = XGBClassifier(
            n_estimators=180,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            num_class=5,
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.feature_extractor = HeartbeatFeatureExtractor(sampling_rate_hz=360)
        self.is_fitted = False
        self.version = "1.0.0"

    def fit(self, X_waveforms: np.ndarray, y_labels: Union[List[str], np.ndarray]) -> "CardiaXClassifier":
        """Extract features, scale, and train the XGBoost classifier.

        Args:
            X_waveforms: 2D array of shape [n_samples, n_samples_per_beat].
            y_labels: 1D array or list of AAMI string labels ('N', 'S', 'V', 'F', 'Q').

        Returns:
            self
        """
        print(f"[1/3] Extracting clinical morphological features from {len(X_waveforms)} heartbeats...")
        X_features = self.feature_extractor.extract_features_matrix(X_waveforms)

        print("[2/3] Standardizing feature distributions...")
        X_scaled = self.scaler.fit_transform(X_features)

        y_encoded = np.array([CLASS_MAPPING[lbl] for lbl in y_labels])

        print("[3/3] Training XGBoost multi-class classifier...")
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True
        print("[OK] CardiaX Model training complete.")
        return self

    def predict_beat(
        self,
        beat_waveform: np.ndarray,
        pre_rr_s: Optional[float] = None,
        post_rr_s: Optional[float] = None,
    ) -> BeatPrediction:
        """Classify a single isolated heartbeat.

        Args:
            beat_waveform: 1D array representing the heartbeat around R-peak.
            pre_rr_s: Preceding RR interval in seconds.
            post_rr_s: Succeeding RR interval in seconds.

        Returns:
            BeatPrediction dataclass instance.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained or loaded before making predictions.")

        feat_dict = self.feature_extractor.extract_single_beat_features(
            beat_waveform, pre_rr_s=pre_rr_s, post_rr_s=post_rr_s
        )
        x_row = np.array([[feat_dict[k] for k in FEATURE_NAMES]])
        x_scaled = self.scaler.transform(x_row)

        probs = self.model.predict_proba(x_scaled)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = INV_CLASS_MAPPING[pred_idx]
        confidence = float(probs[pred_idx])

        prob_dict = {INV_CLASS_MAPPING[i]: float(round(probs[i], 4)) for i in range(5)}

        return BeatPrediction(
            predicted_class=pred_class,
            class_name=CLASS_DESCRIPTIONS[pred_class],
            confidence=float(round(confidence, 4)),
            clinical_risk=CLINICAL_RISK_LEVEL[pred_class],
            probabilities=prob_dict,
            features={k: float(round(v, 4)) for k, v in feat_dict.items()},
        )

    def predict_batch(self, X_waveforms: np.ndarray) -> List[BeatPrediction]:
        """Classify an array of multiple beat waveforms."""
        predictions = []
        for i in range(len(X_waveforms)):
            pred = self.predict_beat(X_waveforms[i])
            predictions.append(pred)
        return predictions

    def save_model(self, file_path: Path | str) -> None:
        """Serialize the complete model package including scaler and metadata."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self.model,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
            "feature_names": FEATURE_NAMES,
            "class_mapping": CLASS_MAPPING,
            "version": self.version,
        }
        joblib.dump(bundle, file_path)
        print(f"[OK] Model bundle saved to: {file_path}")

    @classmethod
    def load_model(cls, file_path: Path | str) -> "CardiaXClassifier":
        """Deserialize and reconstruct a trained CardiaXClassifier instance."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        bundle = joblib.load(file_path)
        instance = cls()
        instance.model = bundle["model"]
        instance.scaler = bundle["scaler"]
        instance.is_fitted = bundle["is_fitted"]
        instance.version = bundle.get("version", "1.0.0")
        return instance
