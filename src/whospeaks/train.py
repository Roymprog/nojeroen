import json
import os

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from importlib.metadata import version as pkg_version

from whospeaks.config import (
    CONFIG_FILENAME,
    EMBEDDING_DIM,
    FEATURE_TYPE,
    MODEL_DIR,
    MODEL_FILENAME,
    SAMPLE_RATE,
    STRIDE_S,
    TARGET_PRECISION,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    WINDOW_SIZE_S,
)
from whospeaks.data_loader import load_split_data


def compute_scale_pos_weight(y):
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return n_neg / max(n_pos, 1)


def train_model(X_train, y_train, scale_pos_weight=None):
    if scale_pos_weight is None:
        scale_pos_weight = compute_scale_pos_weight(y_train)

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model, scale_pos_weight


def tune_threshold(model, X_val, y_val, target_precision=TARGET_PRECISION):
    """Sweep thresholds on validation set to find lowest threshold achieving target precision."""
    val_probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)

    best_threshold = None
    best_recall = 0.0

    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        if prec >= target_precision:
            if best_threshold is None or t < best_threshold:
                best_threshold = t
                best_recall = rec

    if best_threshold is None:
        best_threshold = 0.90
        print(
            f"WARNING: No threshold achieved {target_precision:.0%} precision. "
            f"Using {best_threshold}"
        )

    return best_threshold


def evaluate(model, X, y, threshold):
    """Evaluate model on a dataset at given threshold. Returns metrics dict."""
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "threshold": round(float(threshold), 4),
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "n_predicted_positive": int(preds.sum()),
    }


def run_loso_cv(session_data):
    """Leave-One-Session-Out cross-validation."""
    session_names = list(session_data.keys())
    results = {}

    for held_out in session_names:
        train_sessions = [s for s in session_names if s != held_out]
        X_tr = np.concatenate([session_data[s][0] for s in train_sessions])
        y_tr = np.concatenate([session_data[s][1] for s in train_sessions])
        X_te = session_data[held_out][0]
        y_te = session_data[held_out][1]

        model, _ = train_model(X_tr, y_tr)
        metrics = evaluate(model, X_te, y_te, threshold=0.5)
        results[held_out] = metrics

    return results


def save_model(model, threshold, scale_pos_weight, model_dir=MODEL_DIR):
    """Save model and config to disk."""
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, MODEL_FILENAME)
    joblib.dump({"model": model, "threshold": threshold}, model_path)

    config = {
        "feature_type": FEATURE_TYPE,
        "embedding_dim": EMBEDDING_DIM,
        "window_size_s": WINDOW_SIZE_S,
        "stride_s": STRIDE_S,
        "sample_rate": SAMPLE_RATE,
        "threshold": round(float(threshold), 4),
        "resemblyzer_version": pkg_version("resemblyzer"),
        "scale_pos_weight": round(float(scale_pos_weight), 4),
    }
    config_path = os.path.join(model_dir, CONFIG_FILENAME)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return model_path, config_path


def load_model(model_dir=MODEL_DIR):
    """Load model and threshold from disk."""
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    data = joblib.load(model_path)
    return data["model"], data["threshold"]


def run_training_pipeline(data_dir=None, model_dir=None):
    """Full training pipeline: load data, train, tune, evaluate, save."""
    from whospeaks.config import DATA_DIR as default_data_dir
    from whospeaks.config import MODEL_DIR as default_model_dir

    data_dir = data_dir or default_data_dir
    model_dir = model_dir or default_model_dir

    print("Loading data...")
    split_data, session_data = load_split_data(data_dir=data_dir)
    X_train, y_train, train_counts = split_data["train"]
    X_val, y_val, val_counts = split_data["val"]
    X_test, y_test, test_counts = split_data["test"]

    print(f"Train: {len(y_train)} windows (JEROEN={train_counts['JEROEN_VAN_INKEL']}, OTHER={train_counts['OTHER']})")
    print(f"Val:   {len(y_val)} windows (JEROEN={val_counts['JEROEN_VAN_INKEL']}, OTHER={val_counts['OTHER']})")
    print(f"Test:  {len(y_test)} windows (JEROEN={test_counts['JEROEN_VAN_INKEL']}, OTHER={test_counts['OTHER']})")

    # LOSO-CV
    print("\nRunning LOSO-CV...")
    loso_results = run_loso_cv(
        {s: (session_data[s][0], session_data[s][1]) for s in session_data}
    )
    for fold, metrics in loso_results.items():
        print(f"  {fold}: prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} f1={metrics['f1']:.3f}")

    # Train final model
    print("\nTraining final model...")
    model, spw = train_model(X_train, y_train)

    # Tune threshold
    print("Tuning threshold on validation set...")
    threshold = tune_threshold(model, X_val, y_val)
    print(f"  Tuned threshold: {threshold:.2f}")

    # Evaluate on val and test
    val_metrics = evaluate(model, X_val, y_val, threshold)
    test_metrics = evaluate(model, X_test, y_test, threshold)

    print(f"\nValidation (S3): prec={val_metrics['precision']:.3f} rec={val_metrics['recall']:.3f} f1={val_metrics['f1']:.3f}")
    print(f"Test (S4):       prec={test_metrics['precision']:.3f} rec={test_metrics['recall']:.3f} f1={test_metrics['f1']:.3f}")

    # Save
    print("\nSaving model...")
    model_path, config_path = save_model(model, threshold, spw, model_dir=model_dir)
    print(f"  Model: {model_path}")
    print(f"  Config: {config_path}")

    return {
        "model": model,
        "threshold": threshold,
        "scale_pos_weight": spw,
        "loso_cv": loso_results,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "config_path": config_path,
    }
