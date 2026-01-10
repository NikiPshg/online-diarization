"""
Simple API for speaker diarization.
Supports both PSDA and CSEA backends with overlapping segments and PCA.
"""

import os
from typing import Optional, List, Dict, Tuple
from pyannote.core import Annotation
import sounddevice as sd
import numpy as np
import torch
from queue import Queue
import torchaudio

from onlinediar.cli.psda_diarizate import PsdaSpeakerDiarization
from onlinediar.cli.csea_diarizate import CseaSpeakerDiarization


class Diarizer:
    """
    Simple speaker diarization class.
    
    Usage:
        # PSDA backend
        diarizer = Diarizer(
            backend="psda",
            emb_model_path="models/voxblink2_samresnet34_ft.onnx",
            psda_path="models/psda_vox_128_2sec",
            use_pca=True,
            pca_path="models/pca_model_128d.joblib"
        )
        
        # Register known speakers (optional)
        diarizer.register_speaker("speaker1.wav", "John")
        diarizer.register_speaker("speaker2.wav", "Alice")
        
        # Run diarization on file
        annotation = diarizer.diarize("audio.wav")
        
        # Save to RTTM
        diarizer.save_rttm(annotation, "output.rttm")
        
        # Stream from microphone
        diarizer.stream_from_microphone()
    """

    def __init__(
        self,
        backend: str = "csea",
        emb_model_path: str = None,
        psda_path: Optional[str] = None,
        online_psda_threshold: float = -30.0,
        csea_threshold: float = 0.68,
        max_embeddings_per_speaker: int = 40,
        resample_rate: int = 16000,
        step_size: float = 1.0,
        min_silence_dur_ms: int = 1000,
        min_duration_for_change: float = 1000,
        vad_threshold: float = 0.5,
        duration: float = 2.015,
        device: str = "cpu",
        freeze: bool = False,
        use_pca: bool = False,
        pca_path: Optional[str] = None,
    ):
        """
        Initialize the diarizer.

        Args:
            backend: "psda" or "csea"
            emb_model_path: Path to embedding model (.onnx or directory with .pt model)
            psda_path: Path to PSDA model (required if backend="psda")
            online_psda_threshold: PSDA similarity threshold (lower = stricter)
            csea_threshold: CSEA similarity threshold (higher = stricter)
            max_embeddings_per_speaker: Max embeddings to keep per speaker (CSEA only)
            resample_rate: Audio sample rate
            step_size: Step size for overlapping windows (seconds)
            min_silence_dur_ms: Minimum silence duration (ms)
            min_duration_for_change: Minimum duration to consider speaker change (ms)
            vad_threshold: VAD threshold
            duration: Window duration for embeddings (seconds)
            device: "cpu" or "cuda"
            freeze: If True, only use registered speakers
            use_pca: Use PCA for embedding dimensionality reduction
            pca_path: Path to PCA model (.joblib file)
        """
        self.backend = backend.lower()

        if self.backend == "psda":
            if not psda_path:
                raise ValueError("psda_path is required when using PSDA backend")
            self.model = PsdaSpeakerDiarization(
                emb_model_path=emb_model_path,
                psda_path=psda_path,
                online_psda_threshold=online_psda_threshold,
                resample_rate=resample_rate,
                step_size=step_size,
                min_silence_dur_ms=min_silence_dur_ms,
                min_duration_for_change=min_duration_for_change,
                vad_threshold=vad_threshold,
                duration=duration,
                device=device,
                freeze=freeze,
                use_pca=use_pca,
                pca_path=pca_path,
            )
        elif self.backend == "csea":
            self.model = CseaSpeakerDiarization(
                emb_model_path=emb_model_path,
                csea_threshold=csea_threshold,
                max_embeddings_per_speaker=max_embeddings_per_speaker,
                resample_rate=resample_rate,
                step_size=step_size,
                min_silence_dur_ms=min_silence_dur_ms,
                min_duration_for_change=min_duration_for_change,
                vad_threshold=vad_threshold,
                duration=duration,
                device=device,
                freeze=freeze,
                use_pca=use_pca,
                pca_path=pca_path,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'psda' or 'csea'")

    def diarize(self, audio_path: str) -> Annotation:
        """
        Run diarization on an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            pyannote.core.Annotation object with diarization results
        """
        return self.model.online_diarization(audio_path)

    def register_speaker(self, audio_path: str, speaker_name: str):
        """
        Register a known speaker from an audio file.

        Args:
            audio_path: Path to audio file with speaker's voice
            speaker_name: Name to assign to this speaker
        """
        self.model.recognize(audio_path, speaker_name)

    def reset(self):
        """Reset the diarizer state (clear all speakers and buffers)."""
        self.model.reset()

    @staticmethod
    def save_rttm(annotation: Annotation, output_path: str, audio_uri: str = "audio"):
        """
        Save diarization results to RTTM file.

        Args:
            annotation: pyannote.core.Annotation object
            output_path: Path to output RTTM file
            audio_uri: Audio URI identifier
        """
        with open(output_path, "w") as f:
            for segment, _, speaker in annotation.itertracks(yield_label=True):
                duration = segment.end - segment.start
                f.write(
                    f"SPEAKER {audio_uri} 1 {segment.start:.3f} {duration:.3f} "
                    f"<NA> <NA> {speaker} <NA> <NA>\n"
                )

    def stream_from_microphone(self, save_audio: bool = True, output_prefix: str = "stream_output"):
        """
        Stream audio from microphone and perform real-time diarization.

        Args:
            save_audio: Whether to save recorded audio
            output_prefix: Prefix for output files
        """
        window_size_samples = self.model.window_size_samples
        device = self.model.device
        resample_rate = self.model.resample_rate

        buffer = np.zeros(0, dtype=np.float32)
        full_audio = np.array([])
        speaker_durations = []

        def callback(indata: np.ndarray, *args):
            nonlocal buffer, speaker_durations, full_audio
            try:
                audio_chunk = indata[:, 0]
                
                # Normalize
                max_value = np.max(np.abs(audio_chunk))
                if max_value > 0:
                    audio_chunk = audio_chunk / max_value
                
                buffer = np.concatenate([buffer, audio_chunk])
                full_audio = np.concatenate([full_audio, audio_chunk])

                while len(buffer) >= window_size_samples:
                    chunk = buffer[:window_size_samples]
                    buffer = buffer[window_size_samples:]

                    if len(chunk) == 0:
                        continue

                    chunk_tensor = torch.from_numpy(chunk).float().to(device)
                    
                    # Process chunk through the model's pipeline
                    # Note: This is a simplified streaming version
                    # For full implementation, we'd need to integrate with the run() method
                    
                    print(f"Processing chunk at {len(full_audio) / resample_rate:.2f}s")

            except Exception as e:
                print(f"Callback error: {e}")

        try:
            with sd.InputStream(
                samplerate=resample_rate,
                channels=1,
                dtype="float32",
                blocksize=0,
                latency=None,
                callback=callback,
            ):
                print("🎤 Streaming started. Press Ctrl+C to stop...")
                while True:
                    sd.sleep(1000)

        except KeyboardInterrupt:
            print("\n⏹️  Stopping stream...")
            if save_audio and len(full_audio) > 0:
                audio_path = f"{output_prefix}.wav"
                torchaudio.save(
                    audio_path,
                    torch.from_numpy(full_audio).unsqueeze(0),
                    sample_rate=resample_rate,
                )
                print(f"💾 Audio saved to {audio_path}")
                
                # Run diarization on the recorded audio
                print("🔍 Running diarization on recorded audio...")
                annotation = self.diarize(audio_path)
                rttm_path = f"{output_prefix}.rttm"
                self.save_rttm(annotation, rttm_path)
                print(f"💾 Diarization results saved to {rttm_path}")

    def get_speaker_map(self) -> Dict[str, str]:
        """
        Get the current speaker mapping (speaker_id -> speaker_name).

        Returns:
            Dictionary mapping speaker IDs to names
        """
        return self.model.speaker_map.copy()
