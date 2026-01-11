"""
Real-time streaming diarization examples for PSDA and CSEA backends.
"""

import logging
from onlinediar.diarizer import Diarizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streaming_diarization.log'),
        logging.StreamHandler()
    ]
)


def example_csea_streaming():
    """Example: Real-time streaming with CSEA backend."""
    logger = logging.getLogger(__name__)
    logger.info("=== CSEA Streaming Example ===")
    
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
    
    diarizer.stream_from_microphone(
        save_audio=True,
        output_prefix="csea_stream",
        show_realtime_results=True,
        block_size=4096
    )


def example_psda_streaming():
    """Example: Real-time streaming with PSDA backend."""
    logger = logging.getLogger(__name__)
    logger.info("=== PSDA Streaming Example ===")
    
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
    
    diarizer.stream_from_microphone(
        save_audio=True,
        output_prefix="psda_stream",
        show_realtime_results=True,
        block_size=4096
    )


if __name__ == "__main__":
    # Choose which backend to run
    # Uncomment one of the following:
    
    # example_csea_streaming()
    example_psda_streaming()
