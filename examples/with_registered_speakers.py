"""
Speaker identification with registered speakers for PSDA and CSEA backends.
"""

import logging
from onlinediar.diarizer import Diarizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def register_and_diarize(diarizer: Diarizer, audio_path: str, output_path: str):
    """
    Register speakers and perform diarization with identification.
    
    Args:
        diarizer: Initialized Diarizer instance
        audio_path: Path to audio file
        output_path: Path to output RTTM file
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Registering speakers...")
    speakers = [
        ("speakers/john.wav", "John"),
        ("speakers/alice.wav", "Alice"),
        ("speakers/bob.wav", "Bob")
    ]
    
    for speaker_audio, name in speakers:
        diarizer.register_speaker(speaker_audio, name)
        logger.info(f"Registered: {name}")
    
    speaker_map = diarizer.get_speaker_map()
    logger.info(f"Total registered speakers: {len(speaker_map)}")
    
    logger.info("Starting diarization with speaker identification")
    annotation = diarizer.diarize(audio_path)
    
    diarizer.save_rttm(annotation, output_path)
    logger.info(f"Results saved: {output_path}")
    
    logger.info("Diarization segments:")
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        speaker_name = speaker_map.get(speaker, speaker)
        duration = segment.end - segment.start
        logger.info(f"  {segment.start:.2f}s - {segment.end:.2f}s ({duration:.2f}s): {speaker_name}")


def example_csea_with_speakers():
    """Example: CSEA backend with registered speakers."""
    logger = logging.getLogger(__name__)
    logger.info("=== CSEA with Speaker Identification ===")
    
    diarizer = Diarizer(
        backend="csea",
        emb_model_path="models/voxblink2_samresnet34_ft.onnx",
        csea_threshold=0.68,
        max_embeddings_per_speaker=40,
        resample_rate=16000,
        device="cpu",
        freeze=False,
        use_pca=True,
        pca_path="models/pca_model_128d.joblib",
        log_level=logging.INFO
    )
    
    register_and_diarize(diarizer, "path/to/audio.wav", "output_csea.rttm")


def example_psda_with_speakers():
    """Example: PSDA backend with registered speakers."""
    logger = logging.getLogger(__name__)
    logger.info("=== PSDA with Speaker Identification ===")
    
    diarizer = Diarizer(
        backend="psda",
        emb_model_path="models/voxblink2_samresnet34_ft.onnx",
        psda_path="models/psda_vox_128_2sec",
        online_psda_threshold=-30.0,
        resample_rate=16000,
        device="cpu",
        freeze=False,
        use_pca=True,
        pca_path="models/pca_model_128d.joblib",
        log_level=logging.INFO
    )
    
    register_and_diarize(diarizer, "path/to/audio.wav", "output_psda.rttm")


if __name__ == "__main__":
    example_csea_with_speakers()
    # example_psda_with_speakers()
