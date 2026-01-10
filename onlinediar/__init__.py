"""
Online Speaker Diarization Library

A library for online speaker diarization with PCA, overlap handling, and speaker registration.
"""

from onlinediar.diarizer import Diarizer
from onlinediar.utils.config_loader import load_config, load_diarizer_from_config

__version__ = "2.0.0"
__all__ = ["Diarizer", "load_config", "load_diarizer_from_config"]
