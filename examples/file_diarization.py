"""
File-based speaker diarization examples for PSDA and CSEA backends.
"""

import logging
from pathlib import Path
from onlinediar.diarizer import Diarizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diarization_batch.log'),
        logging.StreamHandler()
    ]
)


def process_audio_file(diarizer: Diarizer, audio_path: str, output_dir: str = "results"):
    """
    Process a single audio file and save diarization results.
    
    Args:
        diarizer: Initialized Diarizer instance
        audio_path: Path to input audio file
        output_dir: Directory for output files
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing: {audio_path}")
    
    annotation = diarizer.diarize(audio_path)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    audio_name = Path(audio_path).stem
    rttm_path = output_path / f"{audio_name}.rttm"
    
    diarizer.save_rttm(annotation, str(rttm_path), audio_uri=audio_name)
    logger.info(f"Results saved: {rttm_path}")
    
    speakers = len(set(segment[2] for segment in annotation.itertracks(yield_label=True)))
    total_duration = sum(
        segment.end - segment.start 
        for segment, _, _ in annotation.itertracks(yield_label=True)
    )
    logger.info(f"Detected {speakers} speakers, total speech: {total_duration:.2f}s")


def example_csea():
    """Example: Batch diarization using CSEA backend."""
    logger = logging.getLogger(__name__)
    logger.info("=== CSEA Backend Example ===")
    
    diarizer = Diarizer(
        backend="csea",
        emb_model_path="models/voxblink2_samresnet34_ft.onnx",
        csea_threshold=0.68,
        max_embeddings_per_speaker=40,
        resample_rate=16000,
        device="cpu",
        use_pca=True,
        pca_path="models/pca_model_128d.joblib",
        log_level=logging.INFO
    )
    
    audio_files = [
        "data/audio1.wav",
        "data/audio2.wav",
        "data/audio3.wav",
    ]
    
    for audio_file in audio_files:
        process_audio_file(diarizer, audio_file, output_dir="results/csea")
        diarizer.reset()


def example_psda():
    """Example: Batch diarization using PSDA backend."""
    logger = logging.getLogger(__name__)
    logger.info("=== PSDA Backend Example ===")
    
    diarizer = Diarizer(
        backend="psda",
        emb_model_path="models/voxblink2_samresnet34_ft.onnx",
        psda_path="models/voxblinkR34_ft/psda_vox_128_2sec",
        online_psda_threshold=-30.0,
        resample_rate=16000,
        device="cpu",
        use_pca=True,
        pca_path="models/pca_model_128d.joblib",
        log_level=logging.INFO
    )
    
    audio_files = [
        "data/audio1.wav",
        "data/audio2.wav",
        "data/audio3.wav",
    ]
    
    for audio_file in audio_files:
        process_audio_file(diarizer, audio_file, output_dir="results/psda")
        diarizer.reset()


if __name__ == "__main__":
    example_csea()
    example_psda()
