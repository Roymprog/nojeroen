"""Backward-compatible re-export of SpeakerPredictor.

The canonical SpeakerPredictor lives in whospeaks.model.
This module re-exports it for backward compatibility with code
that imports from whospeaks.predict.
"""

from whospeaks.model import SpeakerPredictor

__all__ = ["SpeakerPredictor"]
