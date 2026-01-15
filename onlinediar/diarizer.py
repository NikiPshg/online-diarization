"""
Simple API for speaker diarization.
Supports both PSDA and CSEA backends with overlapping segments and PCA.
"""

import os
import logging
import threading
from typing import Optional, List, Dict, Tuple
from pyannote.core import Annotation
try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    _SOUNDDEVICE_AVAILABLE = False
    sd = None
import numpy as np
import torch
from queue import Queue
import torchaudio

from onlinediar.cli.psda_diarizate import PsdaSpeakerDiarization
from onlinediar.cli.csea_diarizate import CseaSpeakerDiarization


class Diarizer:
    """
    Speaker diarization class with support for streaming and file-based processing.
    
    Supports PSDA and CSEA backends with PCA dimensionality reduction and 
    registered speaker identification.
    
    Usage:
        diarizer = Diarizer(
            backend="csea",
            emb_model_path="models/voxblink2_samresnet34_ft.onnx",
            csea_threshold=0.68,
            use_pca=True,
            pca_path="models/pca_model_128d.joblib"
        )
        
        diarizer.register_speaker("speaker1.wav", "John")
        annotation = diarizer.diarize("audio.wav")
        diarizer.save_rttm(annotation, "output.rttm")
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
        log_level: int = logging.INFO,
    ):
        """
        Initialize the diarizer.

        Args:
            backend: Diarization backend ("psda" or "csea")
            emb_model_path: Path to embedding model (.onnx or .pt)
            psda_path: Path to PSDA model (required for PSDA backend)
            online_psda_threshold: PSDA similarity threshold (lower = stricter)
            csea_threshold: CSEA similarity threshold (higher = stricter)
            max_embeddings_per_speaker: Maximum embeddings per speaker (CSEA)
            resample_rate: Audio sample rate in Hz
            step_size: Step size for overlapping windows in seconds
            min_silence_dur_ms: Minimum silence duration in milliseconds
            min_duration_for_change: Minimum duration for speaker change in milliseconds
            vad_threshold: Voice activity detection threshold
            duration: Window duration for embeddings in seconds
            device: Computation device ("cpu" or "cuda")
            freeze: Use only registered speakers if True
            use_pca: Enable PCA dimensionality reduction
            pca_path: Path to PCA model file (.joblib)
            log_level: Logging level
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
        
        self.backend = backend.lower()
        self.logger.info(f"Initializing {self.backend.upper()} diarization backend")

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
        annote = self.model.online_diarization(audio_path)
        self.reset()
        return annote

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

    def stream_from_microphone(
        self, 
        save_audio: bool = True, 
        output_prefix: str = "stream_output",
        show_realtime_results: bool = True,
        block_size: int = 4096
    ):
        """
        Stream audio from microphone and perform real-time diarization.

        Args:
            save_audio: Save recorded audio to file
            output_prefix: Output file prefix
            show_realtime_results: Display real-time results
            block_size: Audio block size for streaming buffer
        """
        if not _SOUNDDEVICE_AVAILABLE:
            raise ImportError(
                "sounddevice is not available. PortAudio library is required for microphone streaming. "
                "Install it with: sudo apt-get install portaudio19-dev libportaudio2"
            )
        resample_rate = self.model.resample_rate
        
        audio_queue = Queue()
        speaker_segments_queue = Queue()
        full_audio = []
        stop_flag = threading.Event()
        
        def diarization_thread():
            try:
                self.model.run(audio_queue, speaker_segments_queue)
            except Exception as e:
                self.logger.error(f"Diarization thread error: {e}")
        
        def output_thread():
            current_speaker = None
            segment_start = None
            
            while not stop_flag.is_set():
                try:
                    result = speaker_segments_queue.get(timeout=0.1)
                    speaker_id = result['speaker_id']
                    start = result['start']
                    end = result['end']
                    similarity = result.get('similarity', 0.0)
                    speaker_name = self.model.speaker_map.get(speaker_id, speaker_id)
                    
                    if show_realtime_results and current_speaker != speaker_name:
                        if current_speaker is not None:
                            duration = start - segment_start
                            self.logger.info(f"Speaker: {current_speaker}, "
                                           f"Duration: {duration:.2f}s")
                        current_speaker = speaker_name
                        segment_start = start
                        self.logger.info(f"Speaker changed: {speaker_name} "
                                       f"(similarity: {similarity:.3f})")
                except:
                    continue
        
        def audio_callback(indata: np.ndarray, frames, time_info, status):
            if status:
                self.logger.warning(f"Audio callback status: {status}")
            
            audio_chunk = indata[:, 0].copy()
            if save_audio:
                full_audio.append(audio_chunk)
            audio_queue.put(audio_chunk)
        
        diar_thread = threading.Thread(target=diarization_thread, daemon=True)
        diar_thread.start()
        
        out_thread = threading.Thread(target=output_thread, daemon=True)
        out_thread.start()
        
        self.logger.info(f"Starting streaming diarization")
        self.logger.info(f"Sample rate: {resample_rate} Hz, Block size: {block_size}")
        self.logger.info(f"Backend: {self.backend.upper()}")
        if self.model.speaker_map:
            self.logger.info(f"Registered speakers: {list(self.model.speaker_map.values())}")
        
        try:
            with sd.InputStream(
                samplerate=resample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                callback=audio_callback,
            ):
                while True:
                    sd.sleep(100)
                    
        except KeyboardInterrupt:
            self.logger.info("Stopping stream...")
            
        finally:
            stop_flag.set()
            audio_queue.put(None)
            diar_thread.join(timeout=2)
            out_thread.join(timeout=2)
            
            if save_audio and len(full_audio) > 0:
                full_audio_array = np.concatenate(full_audio)
                audio_path = f"{output_prefix}.wav"
                torchaudio.save(
                    audio_path,
                    torch.from_numpy(full_audio_array).unsqueeze(0),
                    sample_rate=resample_rate,
                )
                self.logger.info(f"Audio saved: {audio_path}")
                
                self.logger.info("Running full diarization on recorded audio")
                self.reset()
                annotation = self.diarize(audio_path)
                
                rttm_path = f"{output_prefix}.rttm"
                self.save_rttm(annotation, rttm_path)
                self.logger.info(f"RTTM saved: {rttm_path}")
                
                self._print_diarization_stats(annotation)
            
            self.logger.info("Streaming completed")

    def get_speaker_map(self) -> Dict[str, str]:
        """
        Get the current speaker mapping (speaker_id -> speaker_name).

        Returns:
            Dictionary mapping speaker IDs to names
        """
        return self.model.speaker_map.copy()
    
    def _print_diarization_stats(self, annotation: Annotation):
        """
        Log diarization statistics including speaker distribution and durations.
        
        Args:
            annotation: pyannote.core.Annotation with diarization results
        """
        speakers = set()
        total_duration = 0.0
        speaker_durations = {}
        
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            speakers.add(speaker)
            duration = segment.end - segment.start
            total_duration += duration
            
            if speaker not in speaker_durations:
                speaker_durations[speaker] = 0.0
            speaker_durations[speaker] += duration

        self.logger.info("Diarization Statistics:")
        self.logger.info(f"Total speakers: {len(speakers)}")
        self.logger.info(f"Total speech duration: {total_duration:.2f}s")
        
        for speaker in sorted(speaker_durations.keys()):
            duration = speaker_durations[speaker]
            percentage = (duration / total_duration * 100) if total_duration > 0 else 0
            speaker_name = self.model.speaker_map.get(speaker, speaker)
            self.logger.info(f"  {speaker_name}: {duration:.2f}s ({percentage:.1f}%)")
        