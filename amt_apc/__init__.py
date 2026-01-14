"""
AMT-APC: Automatic Piano Cover by Fine-Tuning an Automatic Music Transcription Model

This package provides tools for automatic piano cover generation using
a fine-tuned automatic music transcription model.

Example usage:
    from amt_apc import Pipeline, SVSampler

    pipeline = Pipeline(device="cuda")
    sampler = SVSampler()

    sv = sampler.sample("level2")
    pipeline.wav2midi("input.wav", "output.mid", sv=sv)
"""

from amt_apc.models import Pipeline, Spec2MIDI, load_model, save_model
from amt_apc.data import SVSampler, wav2feature, preprocess_feature

__version__ = "0.1.0"
__all__ = [
    "Pipeline",
    "Spec2MIDI",
    "load_model",
    "save_model",
    "SVSampler",
    "wav2feature",
    "preprocess_feature",
]
