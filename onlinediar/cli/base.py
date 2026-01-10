import os
import logging
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple, Generator

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
from pyannote.core import Annotation, Segment
from silero_vad import VADIterator, get_speech_timestamps, load_silero_vad
import joblib

from onlinediar.models.speaker_model import get_speaker_model
from onlinediar.utils.checkpoint import load_checkpoint
from onlinediar.utils.utils import get_logger, set_seed

class BaseSpeakerDiarization:
    def __init__(self, 
                 emb_model_path: str,
                 resample_rate: int = 16000,
                 step_size: float = 1.0,
                 min_silence_dur_ms: int = 1000,
                 min_duration_for_change: float = 1000,
                 vad_threshold: float = 0.5,
                 duration: float = 2.015,
                 device: str = "cpu",
                 freeze: bool = False,
                 use_pca: bool = False,
                 pca_path: Optional[str] = None):
        set_seed()
        self.logger = get_logger(os.getcwd(), 'diar.log')
        self.emb_model_path = emb_model_path
        self.resample_rate = resample_rate
        self.step_size = step_size
        self.min_silence_dur_ms = min_silence_dur_ms
        self.min_duration_for_change = min_duration_for_change
        self.vad_threshold = vad_threshold
        self.duration = duration  # 200 fr for onnx
        self.device = device
        self.speaker_map = {}  # for naming in diarization
        self.freeze = freeze
        self.use_pca = use_pca
        self.pca_path = pca_path
        if self.freeze:
            self.logger.info('Table update OFF: registered speakers only')
        self._init_audio_buffers()
        self._init_vad()
        self._init_embedding_model()
        if use_pca and pca_path:
            self.logger.info('PCA is used')
            self._init_pca_model()
    
    def _init_embedding_model(self):
        self.logger.info(f"Loading embedding model from: {self.emb_model_path}")
        if self.emb_model_path.endswith('.onnx'):
            self.use_onnx = True
            so = ort.SessionOptions()
            so.inter_op_num_threads = 1
            providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.emb_model_path, sess_options=so, providers=providers)
            self.chunk_samples = int(self.duration * self.resample_rate)
            self.logger.info(f"Embedding model loaded successfully from onnx")
        elif os.path.isdir(self.emb_model_path):
            self.use_onnx = False
            config_path = os.path.join(self.emb_model_path, 'config.yaml')
            model_file = os.path.join(self.emb_model_path, 'avg_model.pt')
            with open(config_path, 'r') as fin:
                configs = yaml.load(fin, Loader=yaml.FullLoader)
            self.emb_model = get_speaker_model(configs['model'])(**configs['model_args'])
            load_checkpoint(self.emb_model, model_file)
            self.emb_model.eval()
            self.emb_model.to(self.device)
            self.chunk_samples = int(self.duration * self.resample_rate)
            self.logger.info(f"Embedding model loaded successfully from pt")
        else:
            raise ValueError(f"Unsupported embedding model format: {self.emb_model_path}")
        
    def _init_audio_buffers(self):
            self.buffer = np.array([], dtype=np.float32)
            self.full_audio = np.array([], dtype=np.float32)
            self.speech_start = None
            self.speech_end = None
            self.next_start = None

    def _compute_fbank(self, wavform: torch.Tensor) -> torch.Tensor:
        feat = kaldi.fbank(wavform,
                           num_mel_bins=80,
                           frame_length=25,
                           frame_shift=10,
                           sample_frequency=self.resample_rate)
        feat = feat - torch.mean(feat, 0)
        return feat.to(torch.float32)

    def _init_vad(self):
        self.vad_model = load_silero_vad(onnx=True)
        self.vad_iter = VADIterator(
            model=self.vad_model,
            sampling_rate=self.resample_rate,
            threshold=self.vad_threshold,
        )
        self.window_size_samples = 512 if self.resample_rate == 16000 else 256
        self.vad_chunk = self.window_size_samples / self.resample_rate
    
    def _init_pca_model(self):
        self.pca_model = joblib.load(self.pca_path) 
        
    def _is_silence(self, start, end):
        return (end - start) >= self.min_silence_dur_ms / 1000
    
    def _is_short(self, start, end):
        return (end - start) < ( self.min_duration_for_change / 1000 )
    
    def _get_start_end(self, speech_dict: Optional[Dict[str, float]]) -> Optional[Tuple[float, float]]:
        current_time = len(self.full_audio) / self.resample_rate
        if speech_dict:
            if 'start' in speech_dict:
                if self.speech_start is not None and self.speech_end is None:
                    self.speech_end = current_time
                self.speech_start = speech_dict['start']
                self.speech_end = None
                self.next_start = speech_dict['start']
            elif 'end' in speech_dict and self.speech_start is not None:
                self.speech_end = speech_dict['end']

        if self.speech_start is not None and self.next_start is not None:
            window_end = min(self.next_start + self.duration, self.speech_end if self.speech_end is not None else float('inf'))
            if window_end > current_time:
                return None
            
            segment_start = self.next_start
            segment_end = window_end
            self.next_start += self.step_size
            
            if self.speech_end is not None and self.next_start >= self.speech_end:
                self.speech_start = None
                self.speech_end = None
                self.next_start = None
                
            return segment_start, segment_end
            
        return None

    def cosine_similarity(self, e1: torch.Tensor, e2:  torch.Tensor) -> float:
        cosine_score = torch.dot(e1, e2) / (torch.norm(e1) * torch.norm(e2))
        cosine_score = cosine_score.item()
        return (cosine_score + 1.0) / 2  # normalize: [-1, 1] => [0, 1]

        
    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.use_onnx:
            current_length = waveform.shape[-1]
            target_length = self.chunk_samples
            if current_length != target_length:
                if current_length < target_length:
                    pad_size = target_length - current_length
                    waveform = torch.nn.functional.pad(
                        waveform, 
                        (0, pad_size), 
                        mode='constant', 
                        value=0
                    )
                else:
                    waveform = waveform[:, :target_length]

        feats = self._compute_fbank(waveform).unsqueeze(0)
        return self.extract_embedding_from_feats(feats)

        
    
    def extract_embedding_from_feats(self, feats: torch.Tensor) -> torch.Tensor:
        if self.use_onnx:
            embedding = self.session.run(
                output_names=['embs'],
                input_feed={'feats': feats.numpy()}
            )[0]
        else:
            with torch.no_grad():
                embedding = self.emb_model(feats.float().to(self.device))

        embedding = (
            torch.Tensor(embedding[-1]) 
            if isinstance(embedding, tuple) 
            else torch.Tensor(embedding)
        ).cpu()

        embedding = embedding.squeeze() if embedding.dim() > 1 else embedding

        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
        
        if self.use_pca:
            embedding = torch.from_numpy(
                self.pca_model.transform(embedding.unsqueeze(0).numpy())[0]
            )

        return embedding

    
    def extract_embedding_from_pcm(self, audio_path: str) -> torch.Tensor:
        wav, sample_rate = torchaudio.load(audio_path)

        if sample_rate != self.resample_rate:
            wav = torchaudio.functional.resample(
                waveform=wav,
                orig_freq=sample_rate,
                new_freq=self.resample_rate
                )
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        segments = get_speech_timestamps(
            wav,
            self.vad_model,
            return_seconds=True
            )
        pcmTotal = torch.Tensor()
        if len(segments) > 0:  # remove all the silence
            for segment in segments:
                start = int(segment['start'] * sample_rate)
                end = int(segment['end'] * sample_rate)
                pcmTemp = wav[0, start:end]
                pcmTotal = torch.cat([pcmTotal, pcmTemp], 0)
            pcm = pcmTotal.unsqueeze(0)
        else:  # all silence, nospeech
            return None
        pcm = pcm.to(torch.float)
        feats = self._compute_fbank(pcm)
        feats = feats.unsqueeze(0)
        feats = feats.to(self.device)

        return self.extract_embedding_from_feats(feats)
    
    def processing_embedding(
            self,
            embedding: torch.Tensor,
            is_short: bool = False,
            is_silence: bool = False,
            freeze: bool = False,
            recognize: bool = False
            ) -> Tuple[str, float]:
        
        raise NotImplementedError("Subclasses must implement this method")
    
    def online_diarization(self, wav_path: str):
        audio_que = Queue()
        speaker_segments_queue = Queue()

        try:
            waveform, sr = torchaudio.load(wav_path)
            if waveform.dim() > 1:
                waveform = torch.mean(waveform, dim=0)
            if sr != self.resample_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.resample_rate)(waveform)
            waveform = waveform.squeeze(0)

            chunk_size = int(self.window_size_samples) 
            for i in range(0, waveform.shape[-1], chunk_size):
                audio_que.put(np.array(waveform[i:i+chunk_size]))
            audio_que.put(None) 
 
            self.run(audio_que, speaker_segments_queue)
            results = []
            while not speaker_segments_queue.empty():
                results.append(speaker_segments_queue.get())
            return self._load_output_annote(list(self._process_diarization_segments(results)))

        except Exception as e:
            self.logger.error(f"Error in online diarization for {wav_path}: {e}")
            return []

    @staticmethod
    def _process_diarization_segments(segments_iterator: List[Dict[str, Any]]) -> Generator[Dict[str, Any], Any, Any]:
        last_segment = None
        
        for current_segment in segments_iterator:
            if last_segment is None:
                last_segment = current_segment
                continue

            if last_segment['end'] > current_segment['start']:
                if last_segment['speaker_id'] == current_segment['speaker_id']:
                    merged_segment = {
                        'speaker_id': last_segment['speaker_id'],
                        'start': last_segment['start'],
                        'end': max(last_segment['end'], current_segment['end']),
                        'is_short': False,  
                        'is_silence': False,
                        'similarity': max(last_segment['similarity'], current_segment['similarity'])
                    }
                    last_segment = merged_segment
                    continue
                else:
                    overlap_midpoint = (current_segment['start'] + last_segment['end']) / 2.0
                    last_segment['end'] = overlap_midpoint
                    current_segment['start'] = overlap_midpoint

            if last_segment['start'] < last_segment['end']:
                yield last_segment

            last_segment = current_segment

        if last_segment and last_segment['start'] < last_segment['end']:
            yield last_segment
            
    @staticmethod
    def _load_output_annote(segments: List[Dict[str, Any]]) -> Annotation:
        annotation = Annotation()
        for segment_data in segments:
            if segment_data['start'] < segment_data['end']: 
                segment = Segment(segment_data['start'], segment_data['end'])
                speaker_label = segment_data['speaker_id']
                annotation[segment] = speaker_label
        return annotation

    def reset(self):
        raise NotImplementedError("Subclasses must implement this method")

    def run(self, audio_que: Queue, speaker_segments_queue : Queue):
        raise NotImplementedError("Subclasses must implement this method")
    
    @staticmethod
    def recognize(audio_path: str, speaker_name: str):
        raise NotImplementedError("Subclasses must implement this method")
    