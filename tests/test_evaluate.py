"""Tests for the evaluation module (evaluate.py).

Covers RFC-007 required evaluation tests:
- evaluate_model() returns correct metrics dict
- compute_cosine_similarity_stats() correctness
- generate_evaluation_report() produces complete report
- generate_embedding_visualization() saves plot file
- run_evaluation_pipeline() end-to-end on real data
"""

import json
import os
import tempfile

import numpy as np
import pytest
from sklearn.metrics import precision_score

from whospeaks.config import DATA_DIR, EMBEDDING_DIM, SESSIONS
from whospeaks.train import train_model


# ============================================================
# Unit: evaluate_model returns correct metrics dict
# ============================================================
class TestEvaluateModel:

    def test_returns_required_keys(self):
        """evaluate_model returns dict with all required metric keys."""
        from whospeaks.evaluate import evaluate_model

        np.random.seed(42)
        X = np.random.randn(80, EMBEDDING_DIM).astype(np.float32)
        y = np.array([1] * 20 + [0] * 60)
        model, _ = train_model(X, y)

        metrics = evaluate_model(model, X, y, threshold=0.5)

        required_keys = [
            "precision", "recall", "f1", "confusion_matrix",
            "threshold", "n_samples", "n_positive", "n_predicted_positive",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_metric_value_ranges(self):
        """Precision, recall, f1 are in [0, 1]."""
        from whospeaks.evaluate import evaluate_model

        np.random.seed(42)
        X = np.random.randn(80, EMBEDDING_DIM).astype(np.float32)
        y = np.array([1] * 20 + [0] * 60)
        model, _ = train_model(X, y)

        metrics = evaluate_model(model, X, y, threshold=0.5)

        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_confusion_matrix_shape(self):
        """Confusion matrix is 2x2 list."""
        from whospeaks.evaluate import evaluate_model

        np.random.seed(42)
        X = np.random.randn(80, EMBEDDING_DIM).astype(np.float32)
        y = np.array([1] * 20 + [0] * 60)
        model, _ = train_model(X, y)

        metrics = evaluate_model(model, X, y, threshold=0.5)
        cm = metrics["confusion_matrix"]

        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2

    def test_confusion_matrix_sums_to_n_samples(self):
        """Confusion matrix entries sum to total number of samples."""
        from whospeaks.evaluate import evaluate_model

        np.random.seed(42)
        X = np.random.randn(80, EMBEDDING_DIM).astype(np.float32)
        y = np.array([1] * 20 + [0] * 60)
        model, _ = train_model(X, y)

        metrics = evaluate_model(model, X, y, threshold=0.5)
        cm = metrics["confusion_matrix"]
        total = sum(sum(row) for row in cm)
        assert total == metrics["n_samples"]

    def test_count_fields_correct_types(self):
        """n_samples, n_positive, n_predicted_positive are ints."""
        from whospeaks.evaluate import evaluate_model

        np.random.seed(42)
        X = np.random.randn(80, EMBEDDING_DIM).astype(np.float32)
        y = np.array([1] * 20 + [0] * 60)
        model, _ = train_model(X, y)

        metrics = evaluate_model(model, X, y, threshold=0.5)

        assert isinstance(metrics["n_samples"], int)
        assert isinstance(metrics["n_positive"], int)
        assert isinstance(metrics["n_predicted_positive"], int)
        assert metrics["n_positive"] == 20
        assert metrics["n_samples"] == 80


# ============================================================
# Unit: compute_cosine_similarity_stats correctness
# ============================================================
class TestCosineSimilarityStats:

    def _make_session_data(self):
        """Create synthetic session data with known structure.

        JEROEN embeddings cluster near [1,0,0,...] and
        OTHER embeddings cluster near [0,1,0,...] so we can
        predict similarity relationships.
        """
        np.random.seed(42)
        data = {}
        for sname in ["S1", "S2", "S3", "S4"]:
            n_j = 10
            n_o = 20
            # JEROEN: cluster near unit vector [1,0,0,...]
            jeroen_base = np.zeros(EMBEDDING_DIM)
            jeroen_base[0] = 1.0
            X_j = np.tile(jeroen_base, (n_j, 1)) + np.random.randn(n_j, EMBEDDING_DIM) * 0.01
            # OTHER: cluster near unit vector [0,1,0,...]
            other_base = np.zeros(EMBEDDING_DIM)
            other_base[1] = 1.0
            X_o = np.tile(other_base, (n_o, 1)) + np.random.randn(n_o, EMBEDDING_DIM) * 0.01

            X = np.concatenate([X_j, X_o]).astype(np.float32)
            y = np.array([1] * n_j + [0] * n_o)
            counts = {"JEROEN_VAN_INKEL": n_j, "OTHER": n_o}
            data[sname] = (X, y, counts)
        return data

    def test_intra_show_similarity_high(self):
        """Intra-show JEROEN similarity should be high for tightly clustered data."""
        from whospeaks.evaluate import compute_cosine_similarity_stats

        data = self._make_session_data()
        stats = compute_cosine_similarity_stats(data)

        assert stats["mean_intra_show_jeroen_similarity"] is not None
        assert stats["mean_intra_show_jeroen_similarity"] > 0.8

    def test_cross_show_similarity_high_for_same_distribution(self):
        """Cross-show JEROEN similarity should be high when all sessions have same distribution."""
        from whospeaks.evaluate import compute_cosine_similarity_stats

        data = self._make_session_data()
        stats = compute_cosine_similarity_stats(data)

        assert stats["cross_show_jeroen_centroid_similarity"] is not None
        assert stats["cross_show_jeroen_centroid_similarity"] > 0.8

    def test_jeroen_vs_other_separation(self):
        """JEROEN vs OTHER centroid similarity should be low (speakers are separated)."""
        from whospeaks.evaluate import compute_cosine_similarity_stats

        data = self._make_session_data()
        stats = compute_cosine_similarity_stats(data)

        assert stats["jeroen_vs_other_centroid_similarity"] is not None
        # Clusters are orthogonal, so similarity should be low
        assert stats["jeroen_vs_other_centroid_similarity"] < 0.5

    def test_intra_greater_than_inter(self):
        """Intra-speaker similarity should exceed JEROEN-vs-OTHER similarity."""
        from whospeaks.evaluate import compute_cosine_similarity_stats

        data = self._make_session_data()
        stats = compute_cosine_similarity_stats(data)

        assert stats["mean_intra_show_jeroen_similarity"] > stats["jeroen_vs_other_centroid_similarity"]


# ============================================================
# Integration: generate_embedding_visualization saves plot
# ============================================================
class TestEmbeddingVisualization:

    def _make_session_data(self, n_per_session=20):
        np.random.seed(42)
        data = {}
        for sname in ["S1", "S2", "S3", "S4"]:
            n = n_per_session
            X = np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
            y = np.array([1] * (n // 4) + [0] * (n - n // 4))
            counts = {"JEROEN_VAN_INKEL": n // 4, "OTHER": n - n // 4}
            data[sname] = (X, y, counts)
        return data

    def test_saves_png_file(self):
        """Visualization saves a PNG file to disk with nonzero size."""
        from whospeaks.evaluate import generate_embedding_visualization

        data = self._make_session_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_viz.png")
            result = generate_embedding_visualization(data, out_path)

            assert os.path.exists(out_path), "PNG file not created"
            assert os.path.getsize(out_path) > 0, "PNG file is empty"
            assert result == out_path

    def test_handles_single_class(self):
        """Visualization doesn't raise when only one class is present."""
        from whospeaks.evaluate import generate_embedding_visualization

        np.random.seed(42)
        data = {}
        for sname in ["S1", "S2", "S3", "S4"]:
            X = np.random.randn(10, EMBEDDING_DIM).astype(np.float32)
            y = np.zeros(10)  # all OTHER
            data[sname] = (X, y, {"JEROEN_VAN_INKEL": 0, "OTHER": 10})

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_viz.png")
            generate_embedding_visualization(data, out_path)
            assert os.path.exists(out_path)

    def test_handles_single_session(self):
        """Visualization doesn't raise when only one session is present."""
        from whospeaks.evaluate import generate_embedding_visualization

        np.random.seed(42)
        data = {
            "S1": (
                np.random.randn(20, EMBEDDING_DIM).astype(np.float32),
                np.array([1] * 5 + [0] * 15),
                {"JEROEN_VAN_INKEL": 5, "OTHER": 15},
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_viz.png")
            generate_embedding_visualization(data, out_path)
            assert os.path.exists(out_path)

    def test_handles_minimal_data(self):
        """Visualization handles very small datasets (< perplexity)."""
        from whospeaks.evaluate import generate_embedding_visualization

        np.random.seed(42)
        data = {
            "S1": (
                np.random.randn(5, EMBEDDING_DIM).astype(np.float32),
                np.array([1, 1, 0, 0, 0]),
                {"JEROEN_VAN_INKEL": 2, "OTHER": 3},
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_viz.png")
            generate_embedding_visualization(data, out_path)
            assert os.path.exists(out_path)


# ============================================================
# Integration: generate_evaluation_report produces complete report
# ============================================================
class TestEvaluationReport:

    def test_report_contains_all_sections(self):
        """generate_evaluation_report produces report with all required sections."""
        from whospeaks.evaluate import generate_evaluation_report

        np.random.seed(42)
        X_train = np.random.randn(100, EMBEDDING_DIM).astype(np.float32)
        y_train = np.array([1] * 30 + [0] * 70)
        model, _ = train_model(X_train, y_train)

        X_val = np.random.randn(40, EMBEDDING_DIM).astype(np.float32)
        y_val = np.array([1] * 10 + [0] * 30)
        X_test = np.random.randn(40, EMBEDDING_DIM).astype(np.float32)
        y_test = np.array([1] * 10 + [0] * 30)

        split_data = {
            "train": (X_train, y_train, {"JEROEN_VAN_INKEL": 30, "OTHER": 70}),
            "val": (X_val, y_val, {"JEROEN_VAN_INKEL": 10, "OTHER": 30}),
            "test": (X_test, y_test, {"JEROEN_VAN_INKEL": 10, "OTHER": 30}),
        }
        session_data = {}
        for sname in ["S1", "S2", "S3", "S4"]:
            n = 25
            X = np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
            y = np.array([1] * 7 + [0] * 18)
            session_data[sname] = (X, y, {"JEROEN_VAN_INKEL": 7, "OTHER": 18})

        loso_results = {
            "S1": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
            "S2": {"precision": 0.85, "recall": 0.75, "f1": 0.80},
            "S3": {"precision": 0.88, "recall": 0.82, "f1": 0.85},
            "S4": {"precision": 0.92, "recall": 0.70, "f1": 0.80},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report = generate_evaluation_report(
                model=model,
                threshold=0.5,
                split_data=split_data,
                session_data=session_data,
                loso_results=loso_results,
                output_dir=tmpdir,
            )

            assert "val_metrics" in report
            assert "test_metrics" in report
            assert "loso_cv" in report
            assert "cosine_similarity" in report
            assert "visualization_path" in report
            assert "report_path" in report

            # Verify report JSON was saved
            assert os.path.exists(report["report_path"])
            with open(report["report_path"]) as f:
                saved_report = json.load(f)
            assert "val_metrics" in saved_report
            assert "test_metrics" in saved_report

            # Verify visualization was saved
            assert os.path.exists(report["visualization_path"])
            assert os.path.getsize(report["visualization_path"]) > 0

    def test_report_without_loso(self):
        """Report works without LOSO results (optional param)."""
        from whospeaks.evaluate import generate_evaluation_report

        np.random.seed(42)
        X = np.random.randn(60, EMBEDDING_DIM).astype(np.float32)
        y = np.array([1] * 15 + [0] * 45)
        model, _ = train_model(X, y)

        split_data = {
            "train": (X, y, {"JEROEN_VAN_INKEL": 15, "OTHER": 45}),
            "val": (X[:20], y[:20], {"JEROEN_VAN_INKEL": 5, "OTHER": 15}),
            "test": (X[20:40], y[20:40], {"JEROEN_VAN_INKEL": 5, "OTHER": 15}),
        }
        session_data = {
            "S1": (X[:15], y[:15], {"JEROEN_VAN_INKEL": 5, "OTHER": 10}),
            "S2": (X[15:30], y[15:30], {"JEROEN_VAN_INKEL": 5, "OTHER": 10}),
            "S3": (X[30:45], y[30:45], {"JEROEN_VAN_INKEL": 5, "OTHER": 10}),
            "S4": (X[45:60], y[45:60], {"JEROEN_VAN_INKEL": 5, "OTHER": 10}),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report = generate_evaluation_report(
                model=model,
                threshold=0.5,
                split_data=split_data,
                session_data=session_data,
                output_dir=tmpdir,
            )
            assert "loso_cv" not in report
            assert "val_metrics" in report
            assert "test_metrics" in report


# ============================================================
# Integration: Full pipeline on real data (slow)
# ============================================================
class TestEvaluationPipelineRealData:

    @pytest.mark.slow
    def test_s4_precision_and_recall_targets(self):
        """
        RFC-007 acceptance criteria: S4 precision >= 90%, recall > 0%.
        Also verifies visualization files exist.
        """
        if not os.path.exists(DATA_DIR):
            pytest.skip("Training data not available")

        from whospeaks.data_loader import load_split_data
        from whospeaks.evaluate import generate_evaluation_report
        from whospeaks.train import run_loso_cv, train_model, tune_threshold

        split_data, session_data = load_split_data()
        X_train, y_train, _ = split_data["train"]
        X_val, y_val, _ = split_data["val"]

        model, _ = train_model(X_train, y_train)
        threshold = tune_threshold(model, X_val, y_val)

        loso_data = {s: (session_data[s][0], session_data[s][1]) for s in session_data}
        loso_results = run_loso_cv(loso_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            report = generate_evaluation_report(
                model=model,
                threshold=threshold,
                split_data=split_data,
                session_data=session_data,
                loso_results=loso_results,
                output_dir=tmpdir,
            )

            # AC: S4 precision >= 90%
            assert report["test_metrics"]["precision"] >= 0.90, (
                f"S4 precision {report['test_metrics']['precision']:.3f} < 90%"
            )
            # AC: S4 recall > 0%
            assert report["test_metrics"]["recall"] > 0.0, "S4 recall is 0%"

            # Visualization exists
            assert os.path.exists(report["visualization_path"])
            assert os.path.getsize(report["visualization_path"]) > 0

            # Per-window predictions show at least some JEROEN predictions
            sample_preds = report.get("s4_sample_predictions", [])
            if sample_preds:
                total_jeroen = sum(sp["n_jeroen_windows"] for sp in sample_preds)
                assert total_jeroen > 0, (
                    "No JEROEN windows predicted in any S4 sample segments"
                )
