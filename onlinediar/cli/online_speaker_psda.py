import logging
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torchaudio
from silero_vad import VADIterator, read_audio
from tqdm import tqdm
from onlinediar.clustering import backend
from onlinediar.clustering.online_clustering import OnlineClusteringMemory
from onlinediar.clustering.PSDA.psda.psdamodel import PSDA

from .speaker import Speaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class OnlineSpeakerPSDA(Speaker):
    def __init__(
        self,
        model_path: str,
        psda_path: str,
        online_vad_threshold: float = 0.5,
        online_psda_threshold: float = -300.0,
        min_silence_duration_vad_ms: float = 350.0,
        min_dur_for_changes: float = 1500.0,
        min_silence_dur: float = 1000,
        device: str = "cpu",
    ):
        super().__init__(model_path)

        self.online_vad_threshold = online_vad_threshold
        self.online_psda_threshold = online_psda_threshold
        self.min_silence_duration_vad_ms = min_silence_duration_vad_ms  # in ms
        self.min_dur_for_changes = min_dur_for_changes  # in ms
        self.device = device

        # for vad model
        self.window_size_samples = 512 if self.resample_rate == 16000 else 256

        # for silence detection
        self.last_end = 10e5
        self.silence_dur = min_silence_dur

        logger.info(
            f"""
            Used Parms:
            online_psda_threshold = {self.online_psda_threshold}
            min_silence_duration_vad_ms = {self.min_silence_duration_vad_ms}
            min_dur_for_changes = {self.min_dur_for_changes}
            online_vad_threshold = {self.online_vad_threshold}
            window_size_samples = {self.window_size_samples}
            silence_dur = {self.silence_dur}
            device = {self.device}
            """
        )

        self.vad_iter = VADIterator(
            self.vad.to(self.device),
            sampling_rate=self.resample_rate,
            min_silence_duration_ms=self.min_silence_duration_vad_ms,
            threshold=self.online_vad_threshold,
        )

        similarity_score = lambda x1, x2: backend.score_psda_many2many(
            x1, x2, PSDA.load(psda_path)
        )
        self.psda = OnlineClusteringMemory(
            similarity_score, self.online_psda_threshold, average_scores=False
        )

    def online_diarization(
        self,
        audio_path: str,
        save: bool = True,
    ) -> List[Tuple[str, float, float, str]]:
        speaker_durations = []

        wav = read_audio(audio_path, sampling_rate=self.resample_rate).to(self.device)
        start, end = None, None
        window_size_samples = self.window_size_samples

        for i in tqdm(range(0, len(wav), window_size_samples)):
            chunk = wav[i : i + window_size_samples]
            is_last_chunk = len(chunk) < window_size_samples
            if is_last_chunk:
                chunk = torch.nn.functional.pad(
                    chunk,
                    (0, window_size_samples - len(chunk)),
                    mode="constant",
                    value=1e-6,
                )

            speech_dict = self.vad_iter(chunk)

            start, end = self.update_start_end(
                speech_dict, wav, start, end, is_last_chunk
            )

            if end is not None and (speech_dict or is_last_chunk):
                ts_start, ts_end, speaker_name = self.process_segment(
                    wav[start:end], start, end
                )
                # print(("unk", ts_start, ts_end, speaker_name))
                speaker_durations.append(("unk", ts_start, ts_end, speaker_name))

        if save:
            rttm_file_path = os.path.splitext(audio_path)[0] + "_psda.rttm"
            self.make_rttm(speaker_durations, rttm_file_path)

        self.vad.reset_states()
        self.psda.reset()
        return speaker_durations

    def diar_from_embed_pt(self, folder_path, save=True):
        from tqdm import tqdm

        speaker_durations = []
        file_data = []

        for embed_file in os.listdir(folder_path):
            if not embed_file.endswith(".pt"):
                continue
            try:
                parts = os.path.splitext(embed_file)[0].split("_")

                start_s = float(parts[-2])
                end_s = float(parts[-1])
                audio_name = "_".join(parts[:-2])
            except (IndexError, ValueError) as e:
                print(f"Error {embed_file}: {e}. Skip.")
                continue

            file_data.append(
                {
                    "path": os.path.join(folder_path, embed_file),
                    "start_s": start_s,
                    "end_s": end_s,
                    "audio_name": audio_name,
                    "file_name": embed_file,
                }
            )

        file_data.sort(key=lambda x: x["start_s"])
        for data in tqdm(file_data):
            embed_path = data["path"]
            start_s = data["start_s"]
            end_s = data["end_s"]
            audio_name = data["audio_name"]

            embedding = torch.load(embed_path)
            is_short = (end_s - start_s) <= self.min_dur_for_changes / 1000
            is_silence = (start_s - self.last_end) >= self.silence_dur / 1000

            # print(end_s - start_s,  self.min_dur_for_changes / 1000, is_short)
            # print(( start_s - self.last_end ), self.silence_dur / 1000, is_silence)
            # print('-' * 40)

            speaker_name = self.psda.processing_one_emb(
                embedding,
                is_short,
                is_silence,
            )

            self.last_end = end_s
            speaker_name = f"SPEAKER_{str(speaker_name - 1).zfill(2)}"
            speaker_durations.append(("unk", start_s, end_s, speaker_name))

        if save and speaker_durations:
            self.make_rttm(speaker_durations, f"{audio_name}_psda.rttm")

        self.vad.reset_states()
        self.psda.reset()
        return speaker_durations

    def streaming_inference(self):
        import sounddevice as sd

        window_size_samples = self.window_size_samples
        device = self.device

        buffer = np.zeros(0, dtype=np.float32)
        start, end = None, None
        full_audio = np.array([])
        speaker_durations = []

        def callback(indata: np.ndarray, *args):
            nonlocal buffer, start, end, speaker_durations, full_audio
            try:
                buffer = np.concatenate([buffer, indata[:, 0]])
                while len(buffer) >= window_size_samples:
                    chunk = buffer[:window_size_samples]
                    buffer = buffer[window_size_samples:]
                    full_audio = np.concatenate([full_audio, chunk])

                    if len(chunk) == 0:
                        continue

                    chunk_tensor = torch.from_numpy(chunk).float().to(device)
                    is_last = len(chunk) < window_size_samples
                    speech_dict = self.vad_iter(chunk_tensor)
                    start, end = self.update_start_end(
                        speech_dict, chunk_tensor, start, end, is_last
                    )

                    if end is not None and (speech_dict or is_last):
                        segment = full_audio[start:end]
                        ts_start = start / self.resample_rate
                        ts_end = end / self.resample_rate
                        speaker_name = self.process_segment_for_streaming(
                            segment, ts_start, ts_end
                        )
                        speaker_durations.append(
                            ("unk", ts_start, ts_end, speaker_name)
                        )
                        print(
                            f"{ts_start:.2f} - {ts_end:.2f} [SPEAKER_{str(speaker_name).zfill(2)}]"
                        )

            except Exception as e:
                print("Callback error:", e)

        try:
            with sd.InputStream(
                samplerate=self.resample_rate,
                channels=1,
                dtype="float32",
                blocksize=0,
                latency=None,
                extra_settings=None,
                callback=callback,
            ):
                print("Стриминг запущен. Нажмите Ctrl+C для остановки.")
                while True:
                    sd.sleep(1000)

        except KeyboardInterrupt:
            print("\nЗавершение стриминга...")
            audio_path = f"stream_output_{len(speaker_durations)}.wav"
            torchaudio.save(
                audio_path,
                torch.from_numpy(full_audio).unsqueeze(0),
                sample_rate=self.resample_rate,
            )
            self.vad.reset_states()
            rttm_file_path = os.path.splitext(audio_path)[0] + "_psda.rttm"
            self.make_rttm(speaker_durations, rttm_file_path)

            self.vad.reset_states()
            self.psda.reset()
            print(
                "Стриминг остановлен. Результаты сохранены в",
                audio_path,
                "и",
                rttm_file_path,
            )

    def process_segment_for_streaming(
        self, segment: np.ndarray, ts_start: float, ts_end: float
    ) -> str:
        segment_tensor = torch.from_numpy(segment).float().to(self.device).unsqueeze(0)
        embedding = self.extract_embedding_from_pcm(segment_tensor, self.resample_rate)
        is_short = (
            segment.shape[-1] / self.resample_rate
        ) <= self.min_dur_for_changes / 1000
        is_silence = (ts_start - self.last_end) >= self.silence_dur / 1000
        # print(segment.shape[-1] / self.resample_rate , self.min_dur_for_changes / 1000, is_short)
        # print(( ts_start - self.last_end ), self.min_dur_for_changes / 1000, is_silence)
        # print('-' * 40)
        self.last_end = ts_end

        speaker_name = self.psda.processing_one_emb(embedding, is_short, is_silence)

        return speaker_name

    def update_start_end(
        self,
        speech_dict: Dict,
        wav: torch.Tensor,
        start: Union[int],
        end: Union[int],
        is_last_chunk: bool,
    ) -> Tuple[Union[int], Union[int]]:
        # Такая структура из-за того, что итератор выдает либо start, либо end.
        # В оф репе такая же проблема: VADIterator не видит последний end.

        if isinstance(speech_dict, dict) and "start" in speech_dict:
            start = speech_dict["start"]
            end = None

        if isinstance(speech_dict, dict) and "end" in speech_dict:
            end = speech_dict["end"] if not is_last_chunk else len(wav)

        if is_last_chunk and end is None:
            end = len(wav)

        return start, end

    def process_segment(
        self, segment: torch.Tensor, start: int, end: int
    ) -> Tuple[float, float, str]:
        if segment.dim() == 1:
            segment = segment.unsqueeze(0)

        embedding = self.extract_embedding_from_pcm(
            segment, sample_rate=self.resample_rate
        )

        is_short = (
            segment.shape[-1] / self.resample_rate
        ) <= self.min_dur_for_changes / 1000
        is_silence = (
            start / self.resample_rate - self.last_end
        ) >= self.silence_dur / 1000

        self.last_end = end / self.resample_rate

        speaker_name = self.psda.processing_one_emb(embedding, is_short, is_silence)
        ts_start = start / self.resample_rate
        ts_end = end / self.resample_rate

        return ts_start, ts_end, f"SPEACKER_{str(speaker_name - 1 ).zfill(2)}"
