from queue import Queue
from typing import Tuple, Optional

import numpy as np
import torch

from onlinediar.cli.base import BaseSpeakerDiarization
from onlinediar.clustering import backend
from onlinediar.clustering.online_clustering import OnlineClusteringMemory
from onlinediar.clustering.PSDA.psda.psdamodel import PSDA


class PsdaSpeakerDiarization(BaseSpeakerDiarization):
    def __init__(
        self,
        emb_model_path: str,
        psda_path: str,
        online_psda_threshold: float = -300.0,
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
        super().__init__(
            emb_model_path=emb_model_path,
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
        self.psda_path = psda_path
        self.online_psda_threshold = online_psda_threshold
        self.psda_model = PSDA.load(self.psda_path)
        similarity_score = lambda x1, x2: backend.score_psda_many2many(
            x1, x2, self.psda_model
        )
        self.psda = OnlineClusteringMemory(
            similarity_score, self.online_psda_threshold, average_scores=False
        )

    def run(self, audio_que: Queue, speaker_segments_queue: Queue):
        try:
            while True:
                indata = audio_que.get()
                if indata is None:
                    break

                max_value = np.max(np.abs(indata))

                if max_value > 0:
                    indata = indata / max_value

                self.buffer = np.concatenate((self.buffer, indata))
                self.full_audio = np.concatenate((self.full_audio, indata))

                while self.buffer.shape[-1] >= self.window_size_samples:

                    chunk = torch.from_numpy(self.buffer[: self.window_size_samples])
                    self.buffer = self.buffer[self.window_size_samples :]
                    speech_dict = self.vad_iter(chunk, return_seconds=True)
                    segment_info = self._get_start_end(speech_dict=speech_dict)

                    if segment_info is None:
                        continue

                    start, end = segment_info

                    is_short = self._is_short(start, end)
                    is_silence = self._is_silence(start, end)
                    #  if not is_short we need to get the current chunk result
                    #  if is_short and and not is_silence we need to get the last chunk
                    #  if is_short and is_silence we need to get the current chunk result
                    if is_silence or not is_short:
                        if not is_short:
                            wave = self.full_audio[
                                int(self.resample_rate * start) : int(
                                    self.resample_rate * end
                                )
                            ]
                        else:
                            # if is_short and is_silence we need extract embedding from last 2.015 seconds (delay ~ 2 seconds)
                            wave = self.full_audio[
                                int(-(start + self.duration) * self.resample_rate) :
                            ]
                            start = start
                            end = min(
                                start + self.duration,
                                len(self.full_audio) / self.resample_rate,
                            )
                            is_short = self._is_short(start, end)
                        wave = torch.from_numpy(wave).unsqueeze(0)
                        embedding = self.extract_embedding(wave)
                    else:
                        embedding = None

                    self.last_end = end

                    if embedding is None:
                        continue

                    speaker_id, similarity = self.processing_embedding(
                        embedding=embedding,
                        is_short=is_short,
                        is_silence=is_silence,
                        freeze=self.freeze,
                    )

                    date_dict = {
                        "speaker_id": speaker_id,
                        "start": start,
                        "end": end,
                        "is_short": is_short,
                        "is_silence": is_silence,
                        "similarity": similarity,
                    }
                    speaker_segments_queue.put(date_dict)
                    self.logger.debug(f"Diarizaton segment: {date_dict}")
        except Exception as e:
            self.logger.error(f"Error in diarization: {e}")
            self.reset()

    def processing_embedding(
        self,
        embedding: torch.Tensor,
        is_short: bool,
        is_silence: bool,
        recognize: bool = False,
        freeze: bool = False,
    ) -> Tuple[str, float]:
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        speaker_name, similarity = self.psda.processing_emb(
            emb=embedding,
            is_short=is_short,
            is_silence=is_silence,
            recognize=recognize,
            freeze=freeze,
        )
        return speaker_name, similarity

    def recognize(self, audio_path: str, speaker_name: str):
        """Register a known speaker from an audio file."""
        emb = self.extract_embedding_from_pcm(audio_path)
        if emb is None:
            self.logger.warning(f"Could not extract embedding from {audio_path}")
            return
        speaker, _ = self.processing_embedding(
            embedding=emb,
            is_short=False,
            is_silence=False,
            recognize=True,
            freeze=self.freeze,
        )
        self.speaker_map[speaker] = speaker_name
        self.logger.info(f"Registered speaker {speaker_name} as {speaker}")

    def reset(self):
        self.last_speaker = "SPEAKER_00"
        self.last_end = -1
        self._init_audio_buffers()
        self._init_vad()
