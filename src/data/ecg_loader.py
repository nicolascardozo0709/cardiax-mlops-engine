"""ECG Data Ingestion and Management Module.

Provides functions to load, cache, and serialize ECG heartbeat datasets
and clinical benchmark strips for training and API evaluation.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from src.data.synthetic_generator import SyntheticECGGenerator


def get_default_data_dir() -> Path:
    """Return the absolute path to the project's data directory."""
    return Path(__file__).resolve().parent.parent.parent / "data"


def generate_and_save_dataset(
    output_path: Path | str | None = None,
    samples_per_class: int = 500,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a high-quality multi-class AAMI dataset and save it as compressed NPZ.

    Args:
        output_path: Target path (defaults to data/cardiax_aami_dataset.npz).
        samples_per_class: Number of single-beat waveforms per class (total = 5 * samples_per_class).
        random_state: Random seed.

    Returns:
        Tuple of (X: np.ndarray, y: np.ndarray).
    """
    if output_path is None:
        output_path = get_default_data_dir() / "cardiax_aami_dataset.npz"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = SyntheticECGGenerator(sampling_rate_hz=360, random_state=random_state)
    X, y = generator.generate_balanced_dataset(samples_per_class=samples_per_class)
    y_arr = np.array(y)

    np.savez_compressed(output_path, X=X, y=y_arr)
    print(f"[OK] Generated {len(X)} ECG heartbeats across {len(generator.AAMI_CLASSES)} classes.")
    print(f"[OK] Saved dataset to: {output_path}")

    return X, y_arr


def load_dataset(
    data_path: Path | str | None = None,
    fallback_generate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the AAMI dataset from disk, generating it on-the-fly if not found.

    Args:
        data_path: Path to dataset .npz file.
        fallback_generate: Whether to generate dataset if file doesn't exist.

    Returns:
        Tuple of (X: np.ndarray, y: np.ndarray).
    """
    if data_path is None:
        data_path = get_default_data_dir() / "cardiax_aami_dataset.npz"
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        if fallback_generate:
            print(f"[INFO] Dataset not found at {data_path}. Generating on the fly...")
            return generate_and_save_dataset(output_path=data_path)
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    return X, y


def get_clinical_benchmark_strip(pattern: str = "bigeminy") -> Dict[str, any]:
    """Generate a standard clinical rhythm strip for API benchmarking or UI demonstration.

    Args:
        pattern: 'normal', 'bigeminy', 'trigeminy', 'couplet', 'svt', or 'mixed'.

    Returns:
        Dictionary with timestamps, raw_signal, clean_reference, r_peaks, annotations, and metadata.
    """
    generator = SyntheticECGGenerator(sampling_rate_hz=360, random_state=101)
    strip = generator.generate_rhythm_strip(duration_s=10.0, heart_rate_bpm=74.0, arrhythmia_pattern=pattern)
    return strip
