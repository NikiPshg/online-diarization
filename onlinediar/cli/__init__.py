"""
CLI modules for speaker diarization.
"""

from onlinediar.cli.base import BaseSpeakerDiarization
from onlinediar.cli.psda_diarizate import PsdaSpeakerDiarization
from onlinediar.cli.csea_diarizate import CseaSpeakerDiarization

__all__ = ["BaseSpeakerDiarization", "PsdaSpeakerDiarization", "CseaSpeakerDiarization"]
