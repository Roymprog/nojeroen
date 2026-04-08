"""Evaluation metrics, reporting, and visualization for the speaker classifier.

Implements RFC-007 evaluation requirements:
- Precision, recall, F1 at tuned threshold on val (S3) and test (S4)
- Confusion matrix for S4
- t-SNE embedding visualization colored by speaker and session
- Cosine similarity analysis (intra-speaker vs inter-speaker, within-show vs cross-show)
- Per-window sample predictions on S4 JEROEN segments
- Full evaluation report
"""

import glob
import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.manifold import TSNE

from whospeaks.config import (
    DATA_DIR,
    EMBEDDING_DIM,
    MODEL_DIR,
    POSITIVE_LABEL,
    SAMPLE_RATE,
    SESSIONS,
    TEST_SESSION,
    VAL_SESSION,
)
from whospeaks.feature_extraction import extract_windows_from_file, parse_label


def evaluate_model(model, X, y, threshold):
    """Evaluate model on a dataset at given threshold.

    Returns dict with precision, recall, f1, confusion_matrix, counts.
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds, labels=[0, 1])

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "threshold": round(float(threshold), 4),
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "n_predicted_positive": int(preds.sum()),
    }


def compute_cosine_similarity_stats(session_data):
    """Compute cosine similarity statistics for JEROEN embeddings.

    Computes:
    - Mean intra-show JEROEN similarity (within S1-S3)
    - Mean cross-show JEROEN similarity (S1-S3 centroid vs S4)
    - Mean JEROEN vs OTHER similarity (to show separation)

    Returns dict with similarity statistics.
    """
    # Collect JEROEN and OTHER embeddings per session
    jeroen_by_session = {}
    other_by_session = {}
    for sname, (X, y, _) in session_data.items():
        jeroen_mask = y == 1
        other_mask = y == 0
        if jeroen_mask.any():
            jeroen_by_session[sname] = X[jeroen_mask]
        if other_mask.any():
            other_by_session[sname] = X[other_mask]

    results = {}

    # Intra-show: mean pairwise cosine sim of JEROEN within each of S1-S3
    intra_sims = []
    for s in ["S1", "S2", "S3"]:
        if s not in jeroen_by_session or len(jeroen_by_session[s]) < 2:
            continue
        embs = jeroen_by_session[s]
        norms = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        sim_matrix = norms @ norms.T
        # Upper triangle (exclude diagonal)
        n = len(norms)
        triu_indices = np.triu_indices(n, k=1)
        intra_sims.extend(sim_matrix[triu_indices].tolist())

    results["mean_intra_show_jeroen_similarity"] = (
        round(float(np.mean(intra_sims)), 4) if intra_sims else None
    )

    # Cross-show: S1-S3 JEROEN centroid vs S4 JEROEN centroid
    s1_s3_jeroen = []
    for s in ["S1", "S2", "S3"]:
        if s in jeroen_by_session:
            s1_s3_jeroen.append(jeroen_by_session[s])

    if s1_s3_jeroen and "S4" in jeroen_by_session:
        centroid_s1_s3 = np.mean(np.concatenate(s1_s3_jeroen), axis=0)
        centroid_s4 = np.mean(jeroen_by_session["S4"], axis=0)
        cross_sim = float(
            np.dot(centroid_s1_s3, centroid_s4)
            / (np.linalg.norm(centroid_s1_s3) * np.linalg.norm(centroid_s4))
        )
        results["cross_show_jeroen_centroid_similarity"] = round(cross_sim, 4)
    else:
        results["cross_show_jeroen_centroid_similarity"] = None

    # JEROEN vs OTHER separation: mean cosine sim between JEROEN centroid and OTHER centroid
    all_jeroen = np.concatenate(list(jeroen_by_session.values())) if jeroen_by_session else None
    all_other = np.concatenate(list(other_by_session.values())) if other_by_session else None

    if all_jeroen is not None and all_other is not None:
        centroid_j = np.mean(all_jeroen, axis=0)
        centroid_o = np.mean(all_other, axis=0)
        sep_sim = float(
            np.dot(centroid_j, centroid_o)
            / (np.linalg.norm(centroid_j) * np.linalg.norm(centroid_o))
        )
        results["jeroen_vs_other_centroid_similarity"] = round(sep_sim, 4)
    else:
        results["jeroen_vs_other_centroid_similarity"] = None

    return results


def generate_embedding_visualization(
    session_data, output_path, perplexity=30, random_state=42
):
    """Generate t-SNE visualization of embeddings colored by speaker and session.

    Saves a PNG with two subplots:
    1. Colored by speaker (JEROEN vs OTHER)
    2. Colored by session (S1-S4)

    Args:
        session_data: dict of session_name -> (X, y, counts)
        output_path: path to save PNG file
        perplexity: t-SNE perplexity parameter
        random_state: random seed for reproducibility
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect all embeddings with metadata
    all_X = []
    all_labels = []
    all_sessions = []
    session_names = list(session_data.keys())

    for sname in session_names:
        X, y, _ = session_data[sname]
        all_X.append(X)
        all_labels.append(y)
        all_sessions.extend([sname] * len(y))

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_labels)

    if len(X_all) < 2:
        # Not enough data to visualize
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "Not enough data for t-SNE", ha="center", va="center")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Adjust perplexity for small datasets (must be strictly < n_samples)
    effective_perplexity = min(perplexity, max(2, len(X_all) - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        max_iter=1000,
    )
    X_2d = tsne.fit_transform(X_all)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: By speaker
    ax1 = axes[0]
    colors_speaker = ["#1f77b4" if label == 0 else "#d62728" for label in y_all]
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=colors_speaker, alpha=0.5, s=10)
    ax1.set_title("Embeddings by Speaker")
    # Legend
    from matplotlib.lines import Line2D

    legend_speaker = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=8, label="JEROEN"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8, label="OTHER"),
    ]
    ax1.legend(handles=legend_speaker, loc="best")
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")

    # Plot 2: By session
    ax2 = axes[1]
    session_colors = {"S1": "#1f77b4", "S2": "#ff7f0e", "S3": "#2ca02c", "S4": "#d62728"}
    colors_session = [session_colors.get(s, "#999999") for s in all_sessions]
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=colors_session, alpha=0.5, s=10)
    ax2.set_title("Embeddings by Session")
    legend_session = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=session_colors[s], markersize=8, label=s)
        for s in session_names
        if s in session_colors
    ]
    ax2.legend(handles=legend_session, loc="best")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")

    fig.suptitle("Speaker Embedding Visualization (256-dim GE2E -> t-SNE 2D)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def get_sample_s4_jeroen_predictions(model, threshold, data_dir=DATA_DIR, n_segments=3):
    """Get per-window predictions on sample S4 JEROEN segments.

    Returns list of dicts, each containing segment filename,
    per-window probabilities, and per-window predictions.
    """
    s4_dir = os.path.join(data_dir, SESSIONS[TEST_SESSION])
    all_files = sorted(glob.glob(os.path.join(s4_dir, "*.wav")))
    jeroen_files = [
        f for f in all_files
        if parse_label(os.path.basename(f)) == POSITIVE_LABEL
    ]

    results = []
    for wf in jeroen_files[:n_segments]:
        embeddings = extract_windows_from_file(wf)
        if not embeddings:
            continue

        X_windows = np.array(embeddings)
        probs = model.predict_proba(X_windows)[:, 1]
        preds = (probs >= threshold).astype(int)

        results.append({
            "filename": os.path.basename(wf),
            "n_windows": len(embeddings),
            "probabilities": [round(float(p), 4) for p in probs],
            "predictions": preds.tolist(),
            "n_jeroen_windows": int(preds.sum()),
            "n_other_windows": int(len(preds) - preds.sum()),
        })

    return results


def generate_evaluation_report(
    model,
    threshold,
    split_data,
    session_data,
    loso_results=None,
    output_dir=MODEL_DIR,
):
    """Generate complete evaluation report.

    Args:
        model: trained LightGBM model
        threshold: tuned classification threshold
        split_data: dict with 'train', 'val', 'test' -> (X, y, counts)
        session_data: dict of session_name -> (X, y, counts)
        loso_results: optional pre-computed LOSO-CV results
        output_dir: directory to write report and visualizations

    Returns:
        dict with full evaluation report
    """
    os.makedirs(output_dir, exist_ok=True)

    report = {}

    # Evaluate on validation (S3) and test (S4)
    X_val, y_val, _ = split_data["val"]
    X_test, y_test, _ = split_data["test"]

    report["val_metrics"] = evaluate_model(model, X_val, y_val, threshold)
    report["test_metrics"] = evaluate_model(model, X_test, y_test, threshold)

    # LOSO-CV results
    if loso_results is not None:
        report["loso_cv"] = loso_results

    # Cosine similarity analysis
    report["cosine_similarity"] = compute_cosine_similarity_stats(session_data)

    # Per-window sample predictions on S4 JEROEN segments
    report["s4_sample_predictions"] = get_sample_s4_jeroen_predictions(
        model, threshold
    )

    # Embedding visualization
    viz_path = os.path.join(output_dir, "embedding_visualization.png")
    generate_embedding_visualization(session_data, viz_path)
    report["visualization_path"] = viz_path

    # Save report JSON
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    report["report_path"] = report_path

    return report


def print_evaluation_report(report):
    """Print a human-readable evaluation report to stdout."""
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    for split_name, key in [("Validation (S3)", "val_metrics"), ("Test (S4)", "test_metrics")]:
        m = report[key]
        print(f"\n{split_name}:")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1:        {m['f1']:.4f}")
        print(f"  Threshold: {m['threshold']:.4f}")
        print(f"  Confusion matrix: {m['confusion_matrix']}")
        print(f"  Samples: {m['n_samples']} (positive: {m['n_positive']}, predicted positive: {m['n_predicted_positive']})")

    if "loso_cv" in report:
        print("\nLOSO-CV Results:")
        for fold, metrics in report["loso_cv"].items():
            print(f"  {fold}: prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} f1={metrics['f1']:.3f}")

    cos = report.get("cosine_similarity", {})
    print("\nCosine Similarity Analysis:")
    print(f"  Mean intra-show JEROEN similarity:     {cos.get('mean_intra_show_jeroen_similarity')}")
    print(f"  Cross-show JEROEN centroid similarity:  {cos.get('cross_show_jeroen_centroid_similarity')}")
    print(f"  JEROEN vs OTHER centroid similarity:    {cos.get('jeroen_vs_other_centroid_similarity')}")

    preds = report.get("s4_sample_predictions", [])
    if preds:
        print(f"\nPer-window predictions on {len(preds)} sample S4 JEROEN segments:")
        for sp in preds:
            print(f"  {sp['filename']}: {sp['n_windows']} windows, "
                  f"{sp['n_jeroen_windows']} JEROEN / {sp['n_other_windows']} OTHER")
            print(f"    Probabilities: {sp['probabilities']}")

    if "visualization_path" in report:
        print(f"\nVisualization saved to: {report['visualization_path']}")
    if "report_path" in report:
        print(f"Full report saved to: {report['report_path']}")

    print("=" * 60)


def run_evaluation_pipeline(model_dir=MODEL_DIR, data_dir=DATA_DIR):
    """Run the full evaluation pipeline.

    Loads trained model and data, generates all metrics and visualizations.
    """
    from whospeaks.data_loader import load_split_data
    from whospeaks.train import load_model, run_loso_cv

    print("Loading model...")
    model, threshold = load_model(model_dir)

    print("Loading data...")
    split_data, session_data = load_split_data(data_dir=data_dir)

    print("Running LOSO-CV...")
    loso_data = {s: (session_data[s][0], session_data[s][1]) for s in session_data}
    loso_results = run_loso_cv(loso_data)

    print("Generating evaluation report...")
    report = generate_evaluation_report(
        model=model,
        threshold=threshold,
        split_data=split_data,
        session_data=session_data,
        loso_results=loso_results,
        output_dir=model_dir,
    )

    print_evaluation_report(report)
    return report
