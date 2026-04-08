import glob
import os

import numpy as np

from whospeaks.config import (
    DATA_DIR,
    POSITIVE_LABEL,
    SESSIONS,
    TEST_SESSION,
    TRAIN_SESSIONS,
    VAL_SESSION,
)
from whospeaks.feature_extraction import extract_windows_from_file, parse_label


def load_session_data(session_name, data_dir=DATA_DIR):
    """Load all segments from a session and extract window-level embeddings.

    Returns (X, y, counts) where:
      X: np.array of shape (n_windows, 256)
      y: np.array of shape (n_windows,) with 1=JEROEN, 0=OTHER
      counts: dict with JEROEN_VAN_INKEL and OTHER window counts
    """
    session_dir_name = SESSIONS[session_name]
    session_dir = os.path.join(data_dir, session_dir_name)
    wav_files = sorted(glob.glob(os.path.join(session_dir, "*.wav")))

    all_embeddings = []
    all_labels = []
    counts = {"JEROEN_VAN_INKEL": 0, "OTHER": 0}

    for wf in wav_files:
        label_str = parse_label(os.path.basename(wf))
        if label_str is None:
            continue
        binary_label = 1 if label_str == POSITIVE_LABEL else 0
        embeddings = extract_windows_from_file(wf)
        for emb in embeddings:
            all_embeddings.append(emb)
            all_labels.append(binary_label)
            if binary_label == 1:
                counts["JEROEN_VAN_INKEL"] += 1
            else:
                counts["OTHER"] += 1

    X = np.array(all_embeddings) if all_embeddings else np.empty((0, 256))
    y = np.array(all_labels) if all_labels else np.empty(0)
    return X, y, counts


def load_split_data(data_dir=DATA_DIR):
    """Load train/val/test data using session-based split.

    Returns dict with keys 'train', 'val', 'test', each containing (X, y, counts).
    Also returns per-session data for LOSO-CV.
    """
    session_data = {}
    for sname in SESSIONS:
        X, y, counts = load_session_data(sname, data_dir=data_dir)
        session_data[sname] = (X, y, counts)

    # Compose train from multiple sessions
    X_train = np.concatenate([session_data[s][0] for s in TRAIN_SESSIONS])
    y_train = np.concatenate([session_data[s][1] for s in TRAIN_SESSIONS])
    train_counts = {}
    for key in ["JEROEN_VAN_INKEL", "OTHER"]:
        train_counts[key] = sum(session_data[s][2][key] for s in TRAIN_SESSIONS)

    split_data = {
        "train": (X_train, y_train, train_counts),
        "val": session_data[VAL_SESSION],
        "test": session_data[TEST_SESSION],
    }

    return split_data, session_data
